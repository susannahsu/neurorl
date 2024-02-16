
from typing import Callable, Dict, NamedTuple, Tuple, Optional

import dataclasses
import dm_env
import functools
import chex
import jax
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt
import rlax
import numpy as np
import wandb

from acme import specs
from acme.agents.jax.r2d2 import actor as r2d2_actor
from acme.jax import networks as networks_lib
from acme.agents.jax import actor_core as actor_core_lib
from acme.wrappers import observation_action_reward
from library.utils import episode_mean
from library.utils import make_episode_mask
from library.utils import expand_tile_dim
from library.utils import Discretizer
from library.muzero_mlps import PredictionMlp
from td_agents import basics

import library.networks as networks
from td_agents.basics import ActorObserver, ActorState
from projects.human_sf.utils import SFObserver as Observer


# [T+1, B, N, A]
TIME_AXIS = 0
BATCH_AXIS = 1
POLICY_AXIS = 2
ACTION_AXIS = 3
CUMULANT_AXIS = 4
LARGE_NEGATIVE = -1e7


@dataclasses.dataclass
class Config(basics.Config):
  eval_task_support: str = "train"  # options:
  nsamples: int = 0  # no samples outside of train vector
  variance: float = 0.1

  final_conv_dim: int = 16
  conv_flat_dim: Optional[int] = 0
  sf_layers : Tuple[int]=(256, 256)
  policy_layers : Tuple[int]=(128, 128)
  combine_policy: str = 'sum'

  num_bins: int = 101
  max_scalar_value: float = 5.0
  min_scalar_value: float = 0.0

  head: str = 'independent'
  shared_ind_head: bool = True
  sf_coeff: float = 1e1
  q_coeff: float = 1e3
  sf_lambda: float = .9
  loss_fn: str = 'qlambda'
  sf_activation: str = 'relu'
  sf_mlp_type: str = 'hk'
  out_init_value: Optional[float] = 0.0
  sum_cumulants: bool = True

def cumulants_from_env(data, online_preds, online_state, target_preds, target_state):
  # [T, B, C]
  cumulants = data.observation.observation['state_features']

  # ASSUMPTION: s' has cumulants for (s,a, r)
  cumulants = cumulants[1:].astype(jnp.float32)
  return cumulants

def sample_gauss(mean: jax.Array,
                 var: float,
                 key,
                 nsamples: int):
  # gaussian (mean=mean, var=.1I)
  # mean = [D]
  assert nsamples >= 0
  if nsamples >= 1:
    mean = jnp.expand_dims(mean, -2) # [1, D]
    samples = jnp.tile(mean, [nsamples, 1])  # [N, D]
    dims = samples.shape # [N, D]
    samples =  samples + jnp.sqrt(var) * jax.random.normal(key, dims)
    samples = samples.astype(mean.dtype)
  else:
    samples = jnp.expand_dims(mean, axis=1) # [N, D]
  return samples

def index_sf(sfs, action):
  index = jax.vmap(rlax.batched_index, in_axes=(2, None), out_axes=1)
  # [B, A, C] --> [B, C]
  sfs = index(sfs, action)
  return sfs

###################################
# Actor
###################################

def epsilon_greedy_sample(
    q_values: jax.Array,
    epsilon: float,
    key,
    action_mask: jax.Array):

  def uniform(rng):
    logits = jnp.where(action_mask, action_mask, LARGE_NEGATIVE)

    return jax.random.categorical(rng, logits)

  def exploit(rng):
    return jnp.argmax(q_values)

  # Generate a random number to decide explore or exploit
  explore = jax.random.uniform(key) < epsilon
  return jax.lax.cond(explore, uniform, exploit, key)

def get_actor_core(
    networks: basics.NetworkFn,
    config: Config,
    evaluation: bool = False,
    extract_q_values = lambda preds: preds.q_values,
  ):
  """Returns ActorCore for R2D2."""

  def select_action(params: networks_lib.Params,
                    observation: networks_lib.Observation,
                    state: ActorState[actor_core_lib.RecurrentState]):
    rng, policy_rng = jax.random.split(state.rng)

    preds, recurrent_state = networks.apply(params, policy_rng, observation, state.recurrent_state, evaluation)

    q_values = extract_q_values(preds)
    # BELOW is very idiosyncratic to env
    action_mask = observation.observation.get('action_mask', jnp.ones_like(q_values))

    rng, q_rng = jax.random.split(rng)
    action = epsilon_greedy_sample(
      q_values=q_values, action_mask=action_mask,
      key=q_rng, epsilon=state.epsilon)

    return action, ActorState(
        rng=rng,
        epsilon=state.epsilon,
        step=state.step + 1,
        predictions=preds,
        recurrent_state=recurrent_state,
        prev_recurrent_state=state.recurrent_state)

  def init(
      rng: networks_lib.PRNGKey
  ) -> ActorState[actor_core_lib.RecurrentState]:
    rng, epsilon_rng, state_rng = jax.random.split(rng, 3)
    if not evaluation:
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

###################################
# Loss functions
###################################

def q_learning_lambda_target(
    r_t: jax.Array,
    discount_t: jax.Array,
    q_t: jax.Array,
    a_t: jax.Array,
    lambda_: jax.Array,
    stop_target_gradients: bool = True,
) -> jax.Array:
  """Calculates Peng's or Watkins' Q(lambda) temporal difference error.

  See "Reinforcement Learning: An Introduction" by Sutton and Barto.
  (http://incompleteideas.net/book/ebook/node78.html).

  Args:
    q_tm1: sequence of Q-values at time t-1.
    a_tm1: sequence of action indices at time t-1.
    r_t: sequence of rewards at time t.
    discount_t: sequence of discounts at time t.
    q_t: sequence of Q-values at time t.
    a_t: action index at times [[1, ... , T]] used to select target q-values to bootstrap from; max(target_q_t) for normal Q-learning, max(q_t) for double Q-learning.
    lambda_: mixing parameter lambda, either a scalar (e.g. Peng's Q(lambda)) or
      a sequence (e.g. Watkin's Q(lambda)).
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.

  Returns:
    Q(lambda) temporal difference error.
  """
  v_t = rlax.batched_index(q_t, a_t)
  target_tm1 = rlax.lambda_returns(r_t, discount_t, v_t, lambda_)

  target_tm1 = jax.lax.select(stop_target_gradients,
                              jax.lax.stop_gradient(target_tm1), target_tm1)
  return target_tm1

def n_step_q_learning_target(
    target_q_t: jax.Array,
    a_t: jax.Array,
    r_t: jax.Array,
    discount_t: jax.Array,
    n: int = 5,
    stop_target_gradients: bool = True,
    tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR,
) -> jax.Array:
  """Calculates transformed n-step TD errors.

  See "Recurrent Experience Replay in Distributed Reinforcement Learning" by
  Kapturowski et al. (https://openreview.net/pdf?id=r1lyTjAqYX).

  Args:
    q_tm1: Q-values at times [0, ..., T - 1].
    a_tm1: action index at times [0, ..., T - 1].
    target_q_t: target Q-values at time [1, ..., T].
    a_t: action index at times [[1, ... , T]] used to select target q-values to
      bootstrap from; max(target_q_t) for normal Q-learning, max(q_t) for double
      Q-learning.
    r_t: reward at times [1, ..., T].
    discount_t: discount at times [1, ..., T].
    n: number of steps over which to accumulate reward before bootstrapping.
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.
    tx_pair: TxPair of value function transformation and its inverse.

  Returns:
    Transformed N-step TD error.
  """
  v_t = rlax.batched_index(target_q_t, a_t)
  target_tm1 = rlax.transformed_n_step_returns(
      tx_pair, r_t, discount_t, v_t, n,
      stop_target_gradients=stop_target_gradients)
  return target_tm1

@dataclasses.dataclass
class SFTargetFn:
  bootstrap_n: int = 5
  loss_fn : str = 'qlearning'

  def __call__(
    self,
    cumulants: jax.Array,
    discounts: jax.Array,
    target_sf: jax.Array,
    lambda_: jax.Array,
    selector_actions: jax.Array,
    ):

    if self.loss_fn == 'qlearning':
      target_fn = functools.partial(
        n_step_q_learning_target,
        n=self.bootstrap_n)
      # vmap over cumulant dimension (C), return in dim=3
      target_fn = jax.vmap(
        target_fn,
        in_axes=(2, None, 1, None), out_axes=1)
      targets = target_fn(
        target_sf,  # [T, A, C]
        selector_actions,  # [T]      
        cumulants,  # [T, C]      
        discounts)  # [T]         

    elif self.loss_fn == 'qlambda':
      target_fn = jax.vmap(q_learning_lambda_target,
        in_axes=(1, None, 2, None, None), out_axes=1)

      targets = target_fn(
        cumulants,         # [T, C]    (vmap 1)
        discounts,         # [T]       (vmap None)
        target_sf,         # [T, A, C] (vmap 2)
        selector_actions,  # [T]      
        lambda_,           # [T]       (vmap None)
      )
    else:
      raise NotImplementedError

    return targets


@dataclasses.dataclass
class UsfaLossFn(basics.RecurrentLossFn):

  discretizer: Discretizer = Discretizer(min_value=0, max_value=5, num_bins=101)
  extract_cumulants: Callable = cumulants_from_env
  extract_task: Callable = lambda data: data.observation.observation['task']
  extract_task_dim: Callable = lambda x: x[:, :, 0]

  sf_coeff: float = 1.0
  q_coeff: float = 1.0
  loss_fn : str = 'qlearning'
  lambda_: float  = .9
  sum_cumulants: bool = True
  indicator_weights: bool = False

  def error(self, data, online_preds, online_state, target_preds, target_state, **kwargs):
    # ======================================================
    # Prepare Data
    # ======================================================
    # all are [T+1, B, N, A, C]
    # N = num policies, A = actions, C = cumulant dim
    online_z = online_preds.policy
    target_sf = target_preds.sf
    # pseudo rewards, [T/T+1, B, C]
    cumulants = self.extract_cumulants(
      data=data, online_preds=online_preds, online_state=online_state,
      target_preds=target_preds, target_state=target_state)
    cumulants = cumulants.astype(online_z.dtype)

    # [T, B]
    episode_mask = make_episode_mask(data, include_final=False)
    target_sf = target_sf*episode_mask[:,:, None, None, None]


    # Get selector actions from online Q-values for double Q-learning.
    selector_actions = jnp.argmax(online_preds.all_q_values, axis=ACTION_AXIS) # [T+1, B, N]

    # Preprocess discounts & rewards.
    discounts = (data.discount * self.discount).astype(online_z.dtype) # [T, B]
    lambda_ = (data.discount * self.lambda_).astype(online_z.dtype) 

    # ======================================================
    # Prepare loss (via vmaps)
    # ======================================================
    def td_targets_fn(
      target_sf_,
      selector_actions_,
      cumulants_,
      discounts_,
      lambda__,
    ):
      return SFTargetFn(bootstrap_n=self.bootstrap_n, loss_fn=self.loss_fn)(
        cumulants_,
        discounts_,
        target_sf_,
        lambda__,
        selector_actions_
      )

    # vmap over batch dimension (B), return B in dim=1
    td_targets_fn = jax.vmap(td_targets_fn, in_axes=BATCH_AXIS, out_axes=BATCH_AXIS)

    # vmap over policy dimension (N), return N in dim=2
    td_targets_fn = jax.vmap(
      td_targets_fn,
      in_axes=(POLICY_AXIS, POLICY_AXIS, None, None, None),
      out_axes=POLICY_AXIS)

    # output = [0=T, 1=B, 2=N, 3=C]
    sf_td_targets = td_targets_fn(
      target_sf[1:],  # [T, B, N, A, C] (vmap 2,1)
      selector_actions[1:],  # [T, B, N]       (vmap 2,1)
      cumulants,  # [T-1, B, C]       (vmap None,1)
      discounts[:-1],  # [T, B]          (vmap None,1)
      lambda_[:-1],
      )

    # [T, B, N, A, C, M]
    online_sf_logits = online_preds.sf_logits
    online_sf_logits = online_sf_logits*episode_mask[:,:, None, None, None, None]

    def compute_loss(targets, logits, actions):
      # targets: [C]
      # logits: [A, C, M]
      # actions: []
      index = lambda x, i: jnp.take_along_axis(x, i, axis=0)

      # [A, C]
      logits_action = index(logits, actions[None, None, None])
      logits_action = jnp.squeeze(logits_action, axis=0)

      # [C] --> [C, M]
      scalar_to_probs = jax.vmap(self.discretizer.scalar_to_probs)
      target_probs = scalar_to_probs(targets)

      # [C]
      loss = jax.vmap(rlax.categorical_cross_entropy)(
        target_probs, logits_action)

      return loss

    compute_loss = jax.vmap(compute_loss, 0, 0)  # time
    compute_loss = jax.vmap(compute_loss, 1, 1)  # batch
    compute_loss = jax.vmap(compute_loss, in_axes=(2, 2, None), out_axes=2)  # policies

    # [T, B, N, C]
    sf_loss = compute_loss(
      sf_td_targets,          # [T, B, N, C]
      online_sf_logits[:-1],  # [T, B, N, A, C, M]
      data.action[:-1]        # [T, B]
    )
    sf_loss = sf_loss.mean(POLICY_AXIS)

    unweighted_loss = sf_loss.sum(-1)

    batch_loss = episode_mean(x=unweighted_loss, mask=episode_mask[:-1])
    if self.q_coeff > 0.0:
      task_w = self.extract_task(data)[:-1]
      task_weighted_loss = (sf_loss*task_w).sum(-1)
      task_weighted_loss = episode_mean(
        x=task_weighted_loss, mask=episode_mask[:-1])
      batch_loss += self.q_coeff*task_weighted_loss

    ###########
    # TD Errors
    ###########
    online_sf = online_preds.sf[:-1:, :, 0]
    online_sf_action = jax.vmap(index_sf, 1, 1)(
      online_sf, data.action[:-1])

    # [T, B, C]
    sf_td_error = sf_td_targets[:, :, 0] - online_sf_action
    sf_td_error = sf_td_error.mean(-1)

    if online_preds.action_mask is not None:
      mask = online_preds.action_mask[:-1,:,None]  # [T, B, 1, A]

      # [T, B, N, D]
      online_sf = online_sf*mask[:,:,:,None]

    metrics = {
      '0.total_loss': batch_loss,
      '0.1.unweighted_loss': unweighted_loss,
      '0.1.task_weighted_loss': task_weighted_loss,
      '2.sf_td_error': jnp.abs(sf_td_error),
      '3.cumulants': cumulants,
      '3.sf_mean': online_sf,
      '3.sf_var': online_sf.var(axis=3),
      }

    return sf_td_error, batch_loss, metrics # [T, B], [B]


###################################
# Architectures
###################################
class USFAPreds(NamedTuple):
  all_q_values: jnp.ndarray  # q-value
  q_values: jnp.ndarray  # q-value
  sf: jnp.ndarray # successor features
  sf_logits: jnp.ndarray # successor features
  policy: jnp.ndarray  # policy vector
  task: jnp.ndarray  # task vector (potentially embedded)
  action_mask: Optional[jax.Array] = None

class MonolithicSfHead(hk.Module):
  def __init__(self,
               layers: Tuple[int],
               num_actions: int,
               state_features_dim: int,
               activation: str = 'relu',
               combine_policy: str = 'sum',
               mlp_type: str='hk',
               out_init_value: float = 0.0,
               policy_layers : Tuple[int]=(32,),
               name: Optional[str] = None):
    super(MonolithicSfHead, self).__init__(name=name)
    raise NotImplementedError

    if out_init_value is None:
      out_init = None
    elif out_init_value == 0.0 or out_init_value < 1e-5:
      out_init = jnp.zeros
    else:
      out_init = hk.initializers.TruncatedNormal(stddev=out_init_value)

    if policy_layers:
      policy_modules = [
        hk.Linear(
          policy_layers[0],
          w_init=hk.initializers.TruncatedNormal(),
          with_bias=False)
      ]
      if len(policy_layers) > 1:
        policy_modules.append(
          hk.nets.MLP(
            policy_layers[1:],
            activation=getattr(jax.nn, activation),
            with_bias=False,
          activate_final=True))
      self.policy_net = hk.Sequential(policy_modules)
    else:
      self.policy_net = lambda x: x

    if mlp_type == 'hk':
      self.sf_net = hk.Sequential([
        hk.nets.MLP(tuple(layers), 
                    activation=getattr(jax.nn, activation),
                    activate_final=True,
                    with_bias=False),
        hk.Linear((num_actions * state_features_dim),
                  w_init=out_init,
                  with_bias=False),
      ])
    elif mlp_type == 'muzero':
      from library.muzero_mlps import PredictionMlp
      self.sf_net = PredictionMlp(
        mlp_layers=tuple(layers),
        num_predictions=num_actions * state_features_dim,
        output_init=jnp.zeros if out_init_value else None,
        activation=activation,
      )
    else: raise NotImplementedError


    self.layers = layers
    self.num_actions = num_actions
    self.state_features_dim = state_features_dim
    self.combine_policy = combine_policy

  def __call__(self,
               sf_input: jnp.ndarray,
               policy: jnp.ndarray,
               task: jnp.ndarray) -> jnp.ndarray:
    """Compute successor features and q-valuesu
    
    Args:
        sf_input (jnp.ndarray): D_1
        policy (jnp.ndarray): C_2
        task (jnp.ndarray): D
    
    Returns:
        jnp.ndarray: 2-D tensor of action values of shape [batch_size, num_actions]
    """
    policy = self.policy_net(policy)

    # dim = sf_input.shape[-1]
    linear = lambda x: hk.Linear(self.layers[0], with_bias=False)(x)
    # out_linear = lambda x: hk.Linear(self.layers[0], with_bias=False)(x)

    if self.combine_policy == 'concat':
      sf_input = jnp.concatenate((sf_input, policy), axis=-1)  # 2D
      # sf_input = jax.nn.relu(out_linear(jax.nn.relu(sf_input)))
    elif self.combine_policy == 'product':
      sf_input = linear(sf_input)*linear(policy)
      # sf_input = jax.nn.relu(out_linear(jax.nn.relu(sf_input)))
    elif self.combine_policy == 'sum':
      sf_input = linear(sf_input)+linear(policy)
      # sf_input = jax.nn.relu(out_linear(jax.nn.relu(sf_input)))
    elif self.combine_policy == 'linear':
      sf_input = linear(sf_input)*linear(policy) + linear(policy)
      # sf_input = jax.nn.relu(out_linear(jax.nn.relu(sf_input))) 
    else:
      raise NotImplementedError

    # [A * C]
    sf = self.sf_net(sf_input)
    # [A, C]
    sf = jnp.reshape(sf, (self.num_actions, self.state_features_dim))

    # [A, C] * [1, C] --> [A, C].sum() --> [A]
    q_values = (sf*task[None]).sum(-1)

    assert q_values.shape[0] == self.num_actions, 'wrong shape'
    return sf, q_values

class IndependentSfHead(hk.Module):
  """
  Independent SF heads help with optimization.

  Inspired by 
  1. Categorical Successor Feature Approximators: https://arxiv.org/abs/2310.15940
  2. Modular Successor Feature Approximators: https://arxiv.org/abs/2301.12305

  """
  def __init__(
      self,
      layers: int,
      num_actions: int,
      state_features_dim: int,
      discretizer: Discretizer,
      activation: str = 'relu',
      shared_head: bool = True,
      compositional_policy: bool = False,
      policy_layers : Tuple[int]=(),
      name: Optional[str] = None,
      **kwargs):
    super(IndependentSfHead, self).__init__(name=name)

    self.discretizer = discretizer
    self.sf_net_factory = lambda: PredictionMlp(
        mlp_layers=tuple(layers),
        num_predictions=num_actions * discretizer.num_bins,
        output_init=jnp.zeros,
        activation=activation,
      )
    self.policy_layers = policy_layers
    self.compositional_policy = compositional_policy
    self.shared_head = shared_head


    if policy_layers:
      def make_policy_net():
        policy_modules = [
          hk.Linear(
            policy_layers[0],
            w_init=hk.initializers.TruncatedNormal(),
            with_bias=False)
        ]
        if len(policy_layers) > 1:
          policy_modules.append(
            hk.nets.MLP(
              policy_layers[1:],
              activation=getattr(jax.nn, activation),
              with_bias=False,
            activate_final=True))
        return hk.Sequential(policy_modules)
      self.policy_net_factory = make_policy_net
    else:
      identity = lambda x: x 
      self.policy_net_factory = lambda: identity

    self.layers = layers
    self.num_actions = num_actions
    self.state_features_dim = state_features_dim

  def __call__(self,
               sf_input: jnp.ndarray,
               policy: jnp.ndarray,
               task: jnp.ndarray) -> jnp.ndarray:
    """Compute successor features and q-valuesu
    
    Args:
        sf_input (jnp.ndarray): D
        policy (jnp.ndarray): C
        task (jnp.ndarray): D
    
    Returns:
        jnp.ndarray: 2-D tensor of action values of shape [batch_size, num_actions]
    """
    ndims = self.state_features_dim
    assert policy.shape[0] == ndims

    layer_size = self.layers[0]
    # linear = lambda x: hk.Linear(layer_size, with_bias=False)(x)
    concat = lambda a, b: jnp.concatenate((a, b))
    concat3 = lambda a, b, c: jnp.concatenate((a, b, c))

    # compute policy representation
    dim_indicator = (jnp.abs(policy) < 1e-5).astype(sf_input.dtype)
    policy_inputs = concat(policy, dim_indicator)
    policy = self.policy_net_factory()(policy_inputs)

    # compute sfs
    if self.shared_head:
      sf_net = self.sf_net_factory()
      cumulant_embds = hk.Embed(
        vocab_size=self.state_features_dim,
        embed_dim=layer_size)(jnp.arange(ndims))  # [C, D]
      sfs = [sf_net(concat3(sf_input, policy, cumulant_embds[i])) for i in range(ndims)]
    else:
      sfs = [self.sf_net_factory()(sf_input) for _ in range(ndims)]

    # [C, A*M]
    sf_logits = jnp.stack(sfs)

    # [C, A, M]
    sf_logits = sf_logits.reshape((ndims, self.num_actions, -1))

    # [C, A, M] --> [A, C, M]
    sf_logits = sf_logits.transpose((1, 0, 2))

    logits_to_scalar = jax.vmap(self.discretizer.logits_to_scalar, 0, 0)  # cumulants
    logits_to_scalar = jax.vmap(logits_to_scalar, 1, 1)  # action

    # [A, C, 1]
    sfs = logits_to_scalar(sf_logits)
    sfs = jnp.squeeze(sfs)  # [A, C]

    q_values = (sfs*task[None]).sum(1)
    assert q_values.shape[0] == self.num_actions, 'wrong shape'

    return sfs, sf_logits, q_values

class SfGpiHead(hk.Module):
  """Universal Successor Feature Approximator GPI head"""
  def __init__(self,
    num_actions: int,
    sf_net : hk.Module,
    nsamples: int=10,
    variance: Optional[float]=0.5,
    eval_task_support: str = 'train', 
    ):
    """Summary
    
    Args:
        num_actions (int): Description
        hidden_size (int, optional): hidden size of SF MLP network
        variance (float, optional): variances of sampling
        nsamples (int, optional): number of policies
        eval_task_support (bool, optional): include eval task in support
    
    Raises:
        NotImplementedError: Description
    """
    super(SfGpiHead, self).__init__()
    self.num_actions = num_actions
    self.var = variance
    self.nsamples = nsamples
    self.eval_task_support = eval_task_support
    self.sf_net = sf_net

  def __call__(self,
    usfa_input: jnp.ndarray,  # memory output (e.g. LSTM)
    task: jnp.ndarray,  # task vector
    ) -> USFAPreds:
    policy = task # 1-1 mapping during training
    # -----------------------
    # policies + embeddings
    # -----------------------
    if self.nsamples > 0:
      # sample N times: [D_w] --> [N+1, D_w]
      policy_samples = sample_gauss(
        mean=policy,
        var=self.var,
        key=hk.next_rng_key(),
        nsamples=self.nsamples)
      # combine samples with the original policy vector
      policy_base = jnp.expand_dims(policy, axis=-2) # [1, D_w]
      policies = jnp.concatenate((policy_base, policy_samples), axis=-2)  # [N+1, D_w]
    else:
      policies = jnp.expand_dims(policy, axis=-2) # [1, D_w]

    return self.sfgpi(
      usfa_input=usfa_input,
      policies=policies,
      task=task)

  def evaluate(self,
    task: jnp.ndarray,  # task vector
    usfa_input: jnp.ndarray,  # memory output (e.g. LSTM)
    train_tasks: jnp.ndarray,  # all train tasks
    ) -> USFAPreds:

    if self.eval_task_support == 'train':
      # [N, D]
      policies = train_tasks

    elif self.eval_task_support == 'eval':
      # [1, D]
      policies = jnp.expand_dims(task, axis=-2)

    elif self.eval_task_support == 'train_eval':
      task_expand = jnp.expand_dims(task, axis=-2)
      # [N+1, D]
      policies = jnp.concatenate((train_tasks, task_expand), axis=-2)
    else:
      raise RuntimeError(self.eval_task_support)

    preds = self.sfgpi(
      usfa_input=usfa_input, policies=policies, task=task)

    return preds

  def sfgpi(self,
    usfa_input: jnp.ndarray,
    policies: jnp.ndarray,
    task: jnp.ndarray) -> USFAPreds:
    """Summary
    
    Args:
        usfa_input (jnp.ndarray): D, typically rnn_output
        policies (jnp.ndarray): N x D
        task (jnp.ndarray): D
    Returns:
        USFAPreds: Description
    """

    sfs, sf_logits, all_q_values = jax.vmap(
      self.sf_net, in_axes=(None, 0, None), out_axes=0)(
        usfa_input,
        policies,
        task)

    # GPI
    # -----------------------
    # [N, A] --> [A]
    q_values = jnp.max(all_q_values, axis=-2)
    num_actions = q_values.shape[-1]

    # [N, D] --> [N, A, D]
    policies_repeated = jnp.repeat(policies[:, None], repeats=num_actions, axis=1)

    return USFAPreds(
      sf=sfs,       # [N, A, D_w]
      sf_logits=sf_logits,
      policy=policies_repeated,         # [N, A, D_w]
      all_q_values=all_q_values,
      q_values=q_values,  # [N, A]
      task=task)         # [D_w]

class UsfaArch(hk.RNNCore):
  """Universal Successor Feature Approximator."""

  def __init__(self,
               torso: networks.OarTorso,
               memory: hk.RNNCore,
               head: SfGpiHead,
               name: str = 'usfa_arch'):
    super().__init__(name=name)
    self._torso = torso
    self._memory = memory
    self._head = head

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [D]
      state: hk.LSTMState,  # [D]
      evaluation: bool = False,
  ) -> Tuple[USFAPreds, hk.LSTMState]:
    torso_outputs = self._torso(inputs)  # [D+A+1]
    # context = inputs.observation['context'].astype(torso_outputs.image.dtype)
    # state_features = inputs.observation['state_features'].astype(torso_outputs.image.dtype)
    memory_input = jnp.concatenate(
      (torso_outputs.image, torso_outputs.action), axis=-1)

    core_outputs, new_state = self._memory(memory_input, state)

    # inputs = jax.tree_map(lambda x:x.astype(jnp.float32), inputs)
    if evaluation:
      predictions = self._head.evaluate(
        task=inputs.observation['task'],
        usfa_input=core_outputs,
        train_tasks=inputs.observation['train_tasks']
        )
    else:
      predictions = self._head(
        usfa_input=core_outputs,
        task=inputs.observation['task'],
      )
    return predictions, new_state

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> hk.LSTMState:
    return self._memory.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[USFAPreds, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""

    torso_outputs = hk.BatchApply(self._torso)(inputs)  # [T, B, D+A+1]
    # context = inputs.observation['context'].astype(torso_outputs.image.dtype)
    # state_features = inputs.observation['state_features'].astype(torso_outputs.image.dtype)
    memory_input = jnp.concatenate(
      (torso_outputs.image, torso_outputs.action), axis=-1)

    core_outputs, new_states = hk.static_unroll(
      self._memory, memory_input, state)

    # treat T,B like this don't exist with vmap
    predictions = jax.vmap(jax.vmap(self._head))(
        core_outputs,                # [T, B, D]
        inputs.observation['task'],  # [T, B]
      )
    return predictions, new_states

def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config) -> networks_lib.UnrollableNetwork:
  """Builds default USFA networks for Minigrid games."""

  num_actions = env_spec.actions.num_values
  state_features_dim = env_spec.observations.observation['state_features'].shape[0]

  def make_core_module() -> UsfaArch:
    vision_torso = networks.BabyAIVisionTorso(
        conv_dim=config.final_conv_dim,
        out_dim=config.conv_flat_dim)

    observation_fn = networks.OarTorso(
      num_actions=num_actions,
      vision_torso=vision_torso,
      output_fn=networks.TorsoOutput,
    )

    if config.head == 'independent':
      SfNetCls = functools.partial(
        IndependentSfHead,
        shared_head=config.shared_ind_head)
    elif config.head == 'monolithic':
      SfNetCls = MonolithicSfHead
    else:
      raise NotImplementedError
    discretizer = Discretizer(
      num_bins=config.num_bins,
      min_value=config.min_scalar_value,
      max_value=config.max_scalar_value,
      tx_pair=config.tx_pair)
    sf_net = SfNetCls(
      layers=config.sf_layers,
      state_features_dim=state_features_dim,
      num_actions=num_actions,
      discretizer=discretizer,
      policy_layers=config.policy_layers,
      combine_policy=config.combine_policy,
      activation=config.sf_activation,
      mlp_type=config.sf_mlp_type,
      out_init_value=config.out_init_value,
      )

    usfa_head = SfGpiHead(
      num_actions=num_actions,
      nsamples=config.nsamples,
      variance=config.variance,
      sf_net=sf_net,
      eval_task_support=config.eval_task_support)

    return UsfaArch(
      torso=observation_fn,
      memory=hk.LSTM(config.state_dim),
      head=usfa_head)

  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)
