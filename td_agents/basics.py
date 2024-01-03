
import sys
import functools
import time
from typing import Callable, Iterator, List, Optional, Union, Sequence, Generic
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple
from absl import logging
from acme.tf import savers

import chex
import dm_env
import dataclasses
import haiku as hk
import collections


import acme
from acme import adders, types
from acme import core
from acme import specs
from acme import types as acme_types

from acme.agents.jax.dqn import learning_lib
from acme.utils import async_utils
from acme.utils import counting
from acme.utils import loggers
from acme.jax import utils
from acme.jax import networks as network_lib
from acme.jax import variable_utils
from acme.agents.jax import r2d2
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax.r2d2 import actor as r2d2_actor
from acme.agents.jax.r2d2 import config as r2d2_config
from acme.agents.jax.r2d2 import learning as r2d2_learning
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.jax import networks as networks_lib
from acme.utils import counting
from acme.utils import loggers





import jax
import optax
import reverb
import numpy as np

import jax
import jax.numpy as jnp
from jax._src.lib import xla_bridge # for clearing cache
import optax
import reverb
import rlax
import tree


_PMAP_AXIS_NAME = 'data'
LossFn = learning_lib.LossFn
TrainingState = learning_lib.TrainingState
ReverbUpdate = learning_lib.ReverbUpdate
LossExtra = learning_lib.LossExtra
Policy = r2d2_actor.R2D2Policy

# Only simple observations & discrete action spaces for now.
Observation = jax.Array

# initializations
ValueInitFn = Callable[[networks_lib.PRNGKey, Observation, hk.LSTMState],
                             networks_lib.Params]

# calling networks
RecurrentStateFn = Callable[[networks_lib.Params], hk.LSTMState]
ValueFn = Callable[[networks_lib.Params, Observation, hk.LSTMState],
                         networks_lib.Value]

@chex.dataclass(frozen=True, mappable_dataclass=False)
class ActorState(Generic[actor_core_lib.RecurrentState]):
  rng: networks_lib.PRNGKey
  recurrent_state: actor_core_lib.RecurrentState
  prev_recurrent_state: actor_core_lib.RecurrentState
  epsilon: Optional[jax.Array] = None
  step: Optional[jax.Array] = None


def isbad(x):
  if np.isnan(x):
    raise RuntimeError(f"NaN")
  elif np.isinf(x):
    raise RuntimeError(f"Inf")

def group_named_values(dictionary: dict, name_fn = None):
  if name_fn is None:
    def name_fn(key): return key.split("/")[0]

  new_dict = collections.defaultdict(list)
  for key, named_values in dictionary.items():
    name = name_fn(key)
    new_dict[name].append(sum(named_values.values()))

  return new_dict

def param_sizes(params):

  sizes = group_named_values(
    dictionary=tree.map_structure(jnp.size, params))

  format = lambda number: f"{number:,}"
  return {k: format(sum(v)) for k,v in sizes.items()}

def overlapping(dict1, dict2):
  return set(dict1.keys()).intersection(dict2.keys())

@dataclasses.dataclass
class Config(r2d2.R2D2Config):
  agent: str = 'agent'

  # Architecture
  state_dim: int = 512

  #----------------
  # Epsilon schedule
  #----------------
  linear_epsilon: bool = False  # whether to use linear or sample from log space for all actors
  epsilon_begin: float = .9
  epsilon_end: float = 0.01
  epsilon_steps: Optional[int] = None

  # value-based action-selection options (distributed)
  evaluation_epsilon: float = 0.01
  # num_epsilons: int = 10
  # epsilon_min: float = .01
  # epsilon_max: float = .9
  # epsilon_base: float = .1
  num_epsilons: int = 256
  epsilon_min: float = 1
  epsilon_max: float = 3
  epsilon_base: float = .1

  #----------------
  # # Learner options
  #----------------
  variable_update_period: int = 400  # how often to update actor
  num_sgd_steps_per_step: int = 1
  seed: int = 1
  discount: float = 0.99
  num_steps: int = 6e6
  max_grad_norm: float = 80.0
  adam_eps: float = 1e-3

  # Replay options
  samples_per_insert_tolerance_rate: float = 0.1
  samples_per_insert: float = 0.0
  min_replay_size: int = 10_000
  max_replay_size: int = 100_000
  batch_size: Optional[int] = 32  # number of batch_elements
  burn_in_length: int = 0  # burn in during learning
  trace_length: Optional[int] = 40  # how long training_batch should be
  sequence_period: Optional[int] = None  # how often to add
  prefetch_size: int = 0
  num_parallel_calls: int = 1

  # Priority options
  importance_sampling_exponent: float = 0.6
  priority_exponent: float = 0.9
  max_priority_weight: float = 0.9

@dataclasses.dataclass
class NetworkFn:
  """Pure functions representing recurrent network components.

  Attributes:
    init: Initializes params.
    forward: Computes Q-values using the network at the given recurrent
      state.
    unroll: Applies the unrolled network to a sequence of 
      observations, for learning.
    initial_state: Recurrent state at the beginning of an episode.
  """
  init: ValueInitFn
  forward: ValueFn
  unroll: ValueFn
  initial_state: RecurrentStateFn
  evaluation: Optional[ValueFn] = None

@dataclasses.dataclass
class RecurrentLossFn(learning_lib.LossFn):
  """R2D2 Learning."""
  discount: float = 0.99
  tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR

  # More than DQN
  max_replay_size: int = 100_000
  store_lstm_state: bool = True
  max_priority_weight: float = 0.9
  bootstrap_n: int = 5
  importance_sampling_exponent: float = 0.0
  priority_weights_aux: bool = False
  priority_use_aux: bool = False

  burn_in_length: int = None
  clip_rewards : bool = False
  max_abs_reward: float = 1.
  loss_coeff: float = 1.
  mask_loss: bool = True

  # auxilliary tasks
  aux_tasks: Union[Callable, Sequence[Callable]]=None

  def error(self,
            data: reverb.ReplaySample,
            online_preds : acme_types.NestedArray,
            online_state: acme_types.NestedArray,
            target_preds : acme_types.NestedArray,
            target_state :  acme_types.NestedArray,
            **kwargs):
    """Summary
    
    Args:
        data (TYPE): Description
        online_preds (Predictions): Description
        online_state (TYPE): Description
        target_preds (Predictions): Description
        target_state (TYPE): Description

    Raises:
        NotImplementedError: Description
    """
    raise NotImplementedError

  def __call__(
      self,
      network: NetworkFn,
      params: networks_lib.Params,
      target_params: networks_lib.Params,
      batch: reverb.ReplaySample,
      key_grad: networks_lib.PRNGKey,
      steps: int=None,
    ) -> Tuple[jnp.DeviceArray, learning_lib.LossExtra]:
    """Calculate a loss on a single batch of data."""

    unroll = network.unroll  # convenience

    if self.clip_rewards:
      data = data._replace(reward=jnp.clip(data.reward, -self.max_abs_reward, self.max_abs_reward))

    # Get core state & warm it up on observations for a burn-in period.
    if self.store_lstm_state:
      # Replay core state.
      online_state = utils.maybe_recover_lstm_type(
            batch.data.extras.get('core_state'))
    else:
      _, batch_size = data.action.shape
      key_grad, key = jax.random.split(key_grad)
      online_state = network.init_recurrent_state(key, batch_size)

    ############
    # if we've stored every single RNN state, only use 0th time-step
    # [B, T, D] --> [B, D]
    ############
    if ((isinstance(online_state, hk.LSTMState) and 
        online_state.hidden.ndim > 2)
        or
        (isinstance(online_state, jnp.ndarray) and 
          online_state.ndim > 2)):
      online_state = jax.tree_map(lambda x: x[:,0], online_state)

    target_state = online_state

    # Convert sample data to sequence-major format [T, B, ...].
    data = utils.batch_to_sequence(batch.data)

    # Maybe burn the core state in.
    burn_in_length = self.burn_in_length
    if burn_in_length:
      burn_obs = jax.tree_map(lambda x: x[:burn_in_length], data.observation)
      key_grad, key1, key2 = jax.random.split(key_grad, 3)
      _, online_state = unroll(params, key1, burn_obs, online_state)
      key_grad, key1, key2 = jax.random.split(key_grad, 3)
      _, target_state = unroll(target_params, key2, burn_obs,
                                     target_state)

    # Only get data to learn on from after the end of the burn in period.
    data = jax.tree_map(lambda seq: seq[burn_in_length:], data)

    # Unroll on sequences to get online and target Q-Values.
    key_grad, key1, key2 = jax.random.split(key_grad, 3)
    online_preds, online_state = unroll(params, key1, data.observation, online_state)
    key_grad, key1, key2 = jax.random.split(key_grad, 3)
    target_preds, target_state = unroll(target_params, key2, data.observation,
                               target_state)

    # ======================================================
    # losses
    # ======================================================
    # -----------------------
    # main loss
    # -----------------------
    # [T-1, B], [B]
    elemwise_error, batch_loss, metrics = self.error(
      data=data,
      online_preds=online_preds,
      online_state=online_state,
      target_preds=target_preds,
      target_state=target_state,
      networks=network,
      params=params,
      target_params=target_params,
      steps=steps,
      key_grad=key_grad)
    batch_loss = self.loss_coeff*batch_loss
    elemwise_error = self.loss_coeff*elemwise_error

    # Importance weighting.
    probs = batch.info.probability
    # [B]
    importance_weights = (1. / (probs + 1e-6)).astype(batch_loss.dtype)
    importance_weights **= self.importance_sampling_exponent
    importance_weights /= jnp.max(importance_weights)


    Cls = lambda x: x.__class__.__name__
    metrics={
      Cls(self) : {
        **metrics,
        # 'loss_main': batch_loss.mean(),
        'z.importance': importance_weights.mean(),
        'z.reward' :data.reward.mean()
        }
      }

    # -----------------------
    # auxilliary tasks
    # -----------------------
    total_aux_scalar_loss = 0.0
    total_aux_batch_loss = jnp.zeros(batch_loss.shape, dtype=batch_loss.dtype)
    total_aux_elem_error = jnp.zeros(elemwise_error.shape, dtype=elemwise_error.dtype)

    if self.aux_tasks:
      for aux_task in self.aux_tasks:
        # does this aux task need a random key?
        kwargs=dict()

        if hasattr(aux_task, 'random') and aux_task.random:
          key_grad, key = jax.random.split(key_grad, 2)
          kwargs['key'] = key

        if aux_task.elementwise:
          aux_elemwise_error, aux_batch_loss, aux_metrics = aux_task(
            data=data,
            online_preds=online_preds,
            online_state=online_state,
            target_preds=target_preds,
            target_state=target_state,
            steps=steps,
            **kwargs)
          total_aux_batch_loss += aux_batch_loss
          total_aux_elem_error += aux_elemwise_error
        else:
          aux_loss, aux_metrics = aux_task(
            data=data,
            online_preds=online_preds,
            online_state=online_state,
            target_preds=target_preds,
            target_state=target_state,
            steps=steps,
            **kwargs)
          total_aux_scalar_loss += aux_loss

        metrics[Cls(aux_task)] = aux_metrics

    # -----------------------
    # mean loss over everything
    # -----------------------
    if self.priority_weights_aux:
      # sum all losses and then weight
      total_batch_loss = total_aux_batch_loss + batch_loss # [B]
      mean_loss = jnp.mean(importance_weights * total_batch_loss) # []
      mean_loss += importance_weights.mean()*total_aux_scalar_loss # []
    else:
      mean_loss = jnp.mean(importance_weights * batch_loss) # []
      mean_loss += total_aux_batch_loss.mean() + total_aux_scalar_loss # []

    if self.aux_tasks:
      metrics[Cls(self)]['loss_w_aux'] = mean_loss

    # -----------------------
    # priorities
    # -----------------------
    # Calculate priorities as a mixture of max and mean sequence errors.
    if self.priority_use_aux:
      total_elemwise_error = elemwise_error + total_aux_elem_error
    else:
      total_elemwise_error = elemwise_error

    abs_td_error = jnp.abs(total_elemwise_error).astype(batch_loss.dtype)
    max_priority = self.max_priority_weight * jnp.max(abs_td_error, axis=0)
    mean_priority = (1 - self.max_priority_weight) * jnp.mean(total_elemwise_error, axis=0)
    priorities = (max_priority + mean_priority)

    metrics = jax.tree_map(lambda x: x.mean(), metrics)
    extra = learning_lib.LossExtra(metrics=metrics, reverb_priorities=priorities)

    return mean_loss, extra

class SGDLearner(learning_lib.SGDLearner):
  """An Acme learner based around SGD on batches.

  This learner currently supports optional prioritized replay and assumes a
  TrainingState as described above.
  """

  def __init__(self,
               network: r2d2_networks.R2D2Networks,
               loss_fn: LossFn,
               optimizer_cnstr: Callable[
                 [Config, networks_lib.Params], optax.GradientTransformation],
               target_update_period: int,
               data_iterator: Optional[Iterator[reverb.ReplaySample]]= None,
               random_key: Optional[networks_lib.PRNGKey] = None,
               replay_client: Optional[reverb.Client] = None,
               replay_table_name: str = adders.reverb.DEFAULT_PRIORITY_TABLE,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None,
               num_sgd_steps_per_step: int = 1,
               grad_period: int = 0,
               initialize: bool = True,
               ):
    """Initialize the SGD learner."""
    self._grad_period = grad_period*num_sgd_steps_per_step
    if self._grad_period > 0:
      logging.warning(f'Logging gradients every {self._grad_period} steps')

    # Internalize the loss_fn with network.
    loss_fn = jax.jit(functools.partial(loss_fn, network))

    # SGD performs the loss, optimizer update and periodic target net update.
    def sgd_step(state: TrainingState,
                 batch: reverb.ReplaySample) -> Tuple[TrainingState, LossExtra]:
      next_rng_key, rng_key = jax.random.split(state.rng_key)
      # Implements one SGD step of the loss and updates training state
      (loss, extra), grads = jax.value_and_grad(loss_fn, has_aux=True)(
          state.params, state.target_params, batch, rng_key, state.steps)

      # Average gradients over pmap replicas before optimizer update.
      grads = jax.lax.pmean(grads, axis_name=_PMAP_AXIS_NAME)

      # Apply the optimizer updates
      updates, new_opt_state = self._optimizer.update(grads, state.opt_state, state.params)
      new_params = optax.apply_updates(state.params, updates)

      # Periodically update target networks.
      steps = state.steps + 1
      if isinstance(target_update_period, float):
        assert (target_update_period >= 0.0 and 
                target_update_period < 1.0), 'incorrect float'
        target_params = optax.incremental_update(new_params, state.target_params,
                                                 target_update_period)
      elif isinstance(target_update_period, int):
        assert target_update_period >= 1
        target_params = optax.periodic_update(new_params, state.target_params,
                                              steps, target_update_period)
      else:
        raise NotImplementedError(type(target_update_period))

      new_training_state = TrainingState(
          params=new_params,
          target_params=target_params,
          opt_state=new_opt_state,
          steps=steps,
          rng_key=next_rng_key)

      extra.metrics.update({
        '0.total_loss': loss,
        '0.grad_norm': optax.global_norm(grads),
        '0.update_norm': optax.global_norm(updates),
        '0.param_norm': optax.global_norm(new_params),
      })

      return new_training_state, extra

    # def postprocess_aux(extra: LossExtra) -> LossExtra:
    #   reverb_priorities = jax.tree_util.tree_map(
    #       lambda a: jnp.reshape(a, (-1, *a.shape[2:])), extra.reverb_priorities)
    #   return extra._replace(
    #       metrics=jax.tree_util.tree_map(jnp.mean, extra.metrics),
    #       reverb_priorities=reverb_priorities)

    # Update replay priorities
    def update_priorities(reverb_priorities: ReverbUpdate) -> None:
      if replay_client is None:
        return
      keys, priorities = tree.map_structure(
          # Fetch array and combine device and batch dimensions.
          lambda x: utils.fetch_devicearray(x).reshape((-1,) + x.shape[2:]),
          (reverb_priorities.keys, reverb_priorities.priorities))
      replay_client.mutate_priorities(
          table=replay_table_name,
          updates=dict(zip(keys, priorities)))

    #####################################
    # Internalise agent components
    #####################################
    self._sgd_step = jax.pmap(
      sgd_step, axis_name=_PMAP_AXIS_NAME)

    self._async_priority_updater = async_utils.AsyncExecutor(update_priorities)
    self._counter = counter or counting.Counter()
    self._optimizer_cnstr = optimizer_cnstr
    self._replay_client = replay_client
    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None
    self._num_sgd_steps_per_step = num_sgd_steps_per_step

    if initialize:
      self._data_iterator = data_iterator
      self._logger = logger or loggers.make_default_logger(
          'learner',
          asynchronous=True,
          serialize_fn=utils.fetch_devicearray,
          time_delta=1.,
          steps_key=self._counter.get_steps_key())

      if num_sgd_steps_per_step > 1:
        raise RuntimeError("check")
        # sgd_step = utils.process_multiple_batches(sgd_step, num_sgd_steps_per_step, postprocess_aux)


      # Initialize the network parameters
      self._state, self._optimizer = self.initialize(
        network=network, random_key=random_key)
      self._current_step = 0

  def initialize(self, network, random_key):
      key_params, key_state = jax.random.split(random_key, 2)
      initial_params = network.init(key_params)
      optimizer = self._optimizer_cnstr(initial_params=initial_params)

      state = TrainingState(
          params=initial_params,
          target_params=initial_params,
          opt_state=optimizer.init(initial_params),
          steps=jnp.array(0),
          rng_key=key_state,
      )
      state = utils.replicate_in_all_devices(state)

      # Log how many parameters the network has.
      sizes = tree.map_structure(jnp.size, initial_params)
      total_params =  sum(tree.flatten(sizes.values()))
      logging.info('Total number of params: %.3g', total_params)
      [logging.info(f"{k}: {v}") for k,v in param_sizes(initial_params).items()]

      return state, optimizer

  def set_optimizer(self, optimizer): self._optimizer = optimizer

  def set_state(self, state):
    self._state = state
    self._current_step = utils.get_from_first_device(self._state.steps)

  def step(self):
    """Takes one SGD step on the learner."""
    with jax.profiler.StepTraceAnnotation('step',
                                          step_num=self._current_step):
      data = next(self._data_iterator)
      state, metrics = self.step_data(data, state)
      self.set_state(state)

      self._logger.write(metrics)

  def step_data(self, prefetching_split, state: TrainingState):
    """Takes one SGD step on the learner."""
    # In this case the host property of the prefetching split contains only
    # replay keys and the device property is the prefetched full original
    # sample. Key is on host since it's uint64 type.
    if hasattr(prefetching_split, 'host'):
      # prioritized data
      reverb_keys = prefetching_split.host
      batch: reverb.ReplaySample = prefetching_split.device
    else:
      # regular data
      batch: reverb.ReplaySample = prefetching_split

    state, extra = self._sgd_step(state, batch)

    if self._replay_client and extra.reverb_priorities is not None:
      reverb_priorities = ReverbUpdate(reverb_keys, extra.reverb_priorities)
      self._async_priority_updater.put(reverb_priorities)

    metrics = utils.get_from_first_device(extra.metrics)

    if self._grad_period and self._state.steps % self._grad_period == 0:
      for k, v in metrics['mean_grad'].items():
        # first val
        metrics['mean_grad'][k] = next(iter(v.values())) 
    else:
      metrics.pop('mean_grad', None)

    # Compute elapsed time.
    timestamp = time.time()
    elapsed = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp
    steps_per_sec = (self._num_sgd_steps_per_step / elapsed) if elapsed else 0

    # Update our counts and record it.
    results = self._counter.increment(
        steps=self._num_sgd_steps_per_step, walltime=elapsed)
    results['steps_per_second'] = steps_per_sec
    results.update(metrics)

    return state, results

  def get_state(self): return self._state

def default_adam_constr(config, **kwargs):
  optimizer_chain = [
      optax.clip_by_global_norm(config.max_grad_norm),
      optax.adam(config.learning_rate, eps=config.adam_eps),
    ]
  return optax.chain(*optimizer_chain)

class BasicActor(core.Actor, Generic[actor_core_lib.State, actor_core_lib.Extras]):
  """A generic actor implemented on top of ActorCore.

  An actor based on a policy which takes observations and outputs actions. It
  also adds experiences to replay and updates the actor weights from the policy
  on the learner.
  """

  def __init__(
      self,
      actor: actor_core_lib.ActorCore[actor_core_lib.State, actor_core_lib.Extras],
      random_key: network_lib.PRNGKey,
      variable_client: Optional[variable_utils.VariableClient],
      adders: Optional[Union[adders.Adder, List[adders.Adder]]] = None,
      jit: bool = True,
      backend: Optional[str] = 'cpu',
      per_episode_update: bool = False
  ):
    """Initializes a feed forward actor.

    Args:
      actor: actor core.
      random_key: Random key.
      variable_client: The variable client to get policy parameters from.
      adder: An adder to add experiences to.
      jit: Whether or not to jit the passed ActorCore's pure functions.
      backend: Which backend to use when jitting the policy.
      per_episode_update: if True, updates variable client params once at the
        beginning of each episode
    """
    self._random_key = random_key
    self._variable_client = variable_client
    if adders and not (isinstance(adders, List) or isinstance(adders, Tuple)):
      adders = [adders]

    self._adders = adders
    self._state = None

    # Unpack ActorCore, jitting if requested.
    if jit:
      self._init = jax.jit(actor.init, backend=backend)
      self._policy = jax.jit(actor.select_action, backend=backend)
    else:
      self._init = actor.init
      self._policy = actor.select_action
    self._get_extras = actor.get_extras
    self._per_episode_update = per_episode_update

  @property
  def _params(self):
    return self._variable_client.params if self._variable_client else []

  def select_action(self,
                    observation: network_lib.Observation) -> types.NestedArray:
    action, self._state = self._policy(self._params, observation, self._state)
    return utils.to_numpy(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._random_key, key = jax.random.split(self._random_key)
    self._state = self._init(key)
    if self._adders:
      for adder in self._adders:
        adder.add_first(timestep)
    if self._variable_client and self._per_episode_update:
      self._variable_client.update_and_wait()

  def observe(self, action: network_lib.Action, next_timestep: dm_env.TimeStep):
    if self._adders:
      for adder in self._adders:
        adder.add(
            action, next_timestep, extras=self._get_extras(self._state))

  def update(self, wait: bool = False):
    if self._variable_client and not self._per_episode_update:
      self._variable_client.update(wait)

class Builder(r2d2.R2D2Builder):
  """TD agent Builder. Agent is derivative of R2D2 but may use different network/loss function
  """
  def __init__(self,
              #  networks: r2d2_networks.R2D2Networks,
               config: r2d2_config.R2D2Config,
               LossFn: learning_lib.LossFn=RecurrentLossFn,
               ActorCls: BasicActor = BasicActor,
               optimizer_cnstr = None,
               get_actor_core_fn = None,
               learner_kwargs=None):
    if config.sequence_period is None:
      config.sequence_period = config.trace_length

    super().__init__(config=config)
    if optimizer_cnstr is None:
      optimizer_cnstr = default_adam_constr
    self.optimizer_cnstr = optimizer_cnstr
    self.ActorCls = ActorCls

    if get_actor_core_fn is None:
      get_actor_core_fn = get_actor_core
    self._get_actor_core_fn = get_actor_core_fn
    self._loss_fn = LossFn

    self.learner_kwargs = learner_kwargs or dict()

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: r2d2_networks.R2D2Networks,
      dataset: Iterator[r2d2_learning.R2D2ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec
    # The learner updates the parameters (and initializes them).
    logger = logger_fn('learner')

    return SGDLearner(
        network=networks,
        random_key=random_key,
        optimizer_cnstr=functools.partial(self.optimizer_cnstr,
          config=self._config),
        target_update_period=self._config.target_update_period,
        data_iterator=dataset,
        loss_fn=self._loss_fn,
        replay_client=replay_client,
        replay_table_name=self._config.replay_table_name,
        counter=counter,
        num_sgd_steps_per_step=self._config.num_sgd_steps_per_step,
        logger=logger,
        **self.learner_kwargs)

  def make_policy(self,
                  networks: r2d2_networks.R2D2Networks,
                  environment_spec: specs.EnvironmentSpec,
                  evaluation: bool = False) -> r2d2_actor.R2D2Policy:
    del environment_spec
    return self._get_actor_core_fn(
      networks=networks,
      evaluation=evaluation,
      config=self._config)

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: r2d2_actor.R2D2Policy,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> acme.Actor:
    del environment_spec
    # Create variable client.
    variable_client = variable_utils.VariableClient(
        variable_source,
        key='actor_variables',
        update_period=self._config.variable_update_period)

    return self.ActorCls(
        policy, random_key, variable_client, adder, backend='cpu')

def get_actor_core(
    networks: r2d2_networks.R2D2Networks,
    config: Config,
    evaluation: bool = False,
    linear_epsilon: bool = True,
    extract_q_values = lambda preds: preds
) -> r2d2_actor.R2D2Policy:
  """Returns ActorCore for R2D2."""

  epsilon_schedule = optax.linear_schedule(
          init_value=config.epsilon_begin,
          end_value=config.epsilon_end,
          transition_steps=config.epsilon_steps or config.num_steps//2,
      )
  def select_action(params: networks_lib.Params,
                    observation: networks_lib.Observation,
                    state: ActorState[actor_core_lib.RecurrentState]):
    rng, policy_rng = jax.random.split(state.rng)

    preds, recurrent_state = networks.apply(params, policy_rng, observation, state.recurrent_state)

    q_values = extract_q_values(preds)
    if linear_epsilon:
      epsilon = epsilon_schedule(state.step)
    else:
      epsilon = state.epsilon

    action = rlax.epsilon_greedy(epsilon).sample(policy_rng, q_values)
    return action, ActorState(
        rng=rng,
        epsilon=state.epsilon,
        step=state.step + 1,
        recurrent_state=recurrent_state,
        prev_recurrent_state=state.recurrent_state)

  def init(
      rng: networks_lib.PRNGKey
  ) -> ActorState[actor_core_lib.RecurrentState]:
    rng, epsilon_rng, state_rng = jax.random.split(rng, 3)
    if not evaluation:
      if linear_epsilon:
        epsilon = config.epsilon_begin
      else:
        epsilon = jax.random.choice(epsilon_rng,
                                    np.logspace(
                                      start=config.epsilon_min,
                                      stop=config.epsilon_max,
                                      num=config.num_epsilons,
                                      base=config.epsilon_base))
    else:
      epsilon = config.evaluation_epsilon
    initial_core_state = networks.init_recurrent_state(state_rng, None)
    return ActorState(
        rng=rng,
        epsilon=epsilon,
        step=0,
        recurrent_state=initial_core_state,
        prev_recurrent_state=initial_core_state)

  def get_extras(
      state: ActorState[actor_core_lib.RecurrentState]
  ) -> r2d2_actor.R2D2Extras:
    return {'core_state': state.prev_recurrent_state}

  return actor_core_lib.ActorCore(init=init, select_action=select_action,
                                  get_extras=get_extras)

def disable_insert_blocking(
    tables: Sequence[reverb.Table]
) -> Tuple[Sequence[reverb.Table], Sequence[int]]:
  """Disables blocking of insert operations for a given collection of tables."""
  modified_tables = []
  sample_sizes = []
  for table in tables:
    rate_limiter_info = table.info.rate_limiter_info
    rate_limiter = reverb.rate_limiters.RateLimiter(
        samples_per_insert=rate_limiter_info.samples_per_insert,
        min_size_to_sample=rate_limiter_info.min_size_to_sample,
        min_diff=rate_limiter_info.min_diff,
        max_diff=sys.float_info.max)
    modified_tables.append(table.replace(rate_limiter=rate_limiter))
    # Target the middle of the rate limiter's insert-sample balance window.
    sample_sizes.append(
        max(1, int(
            (rate_limiter_info.max_diff - rate_limiter_info.min_diff) / 2)))
  return modified_tables, sample_sizes

