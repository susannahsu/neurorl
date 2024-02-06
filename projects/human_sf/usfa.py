
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

from td_agents import basics

import library.networks as networks
from td_agents.basics import ActorObserver, ActorState

# [T+1, B, N, A]
TIME_AXIS = 0
BATCH_AXIS = 1
POLICY_AXIS = 2
ACTION_AXIS = 3
CUMULANT_AXIS = 4


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

  head: str = 'independent'
  sf_coeff: float = 1e1
  q_coeff: float = 1.0
  sf_lambda: float = .9
  sf_loss: str = 'qlambda'
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

def cumulants_from_preds(
  data,
  online_preds,
  online_state,
  target_preds,
  target_state,
  stop_grad=True,
  use_target=False):

  if use_target:
    cumulants = target_preds.state_feature
  else:
    cumulants = online_preds.state_feature
  if stop_grad:
    return jax.lax.stop_gradient(cumulants) # [T, B, C]
  else:
    return cumulants # [T, B, C]

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

def get_actor_core(
    networks: basics.NetworkFn,
    config: Config,
    evaluation: bool = False,
    extract_q_values = lambda preds: preds.q_values
  ):
  """Returns ActorCore for R2D2."""

  def select_action(params: networks_lib.Params,
                    observation: networks_lib.Observation,
                    state: ActorState[actor_core_lib.RecurrentState]):
    rng, policy_rng = jax.random.split(state.rng)

    preds, recurrent_state = networks.apply(params, policy_rng, observation, state.recurrent_state, evaluation)

    q_values = extract_q_values(preds)

    epsilon = state.epsilon
    action = rlax.epsilon_greedy(epsilon).sample(policy_rng, q_values)
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

def q_learning_lambda(
    q_tm1: jax.Array,
    a_tm1: jax.Array,
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
  chex.assert_rank([q_tm1, a_tm1, a_t, r_t, discount_t, q_t, lambda_],
                   [2, 1, 1, 1, 1, 2, {0, 1}])
  chex.assert_type([q_tm1, a_tm1, r_t, discount_t, q_t, lambda_],
                   [float, int, float, float, float, float])

  qa_tm1 = rlax.batched_index(q_tm1, a_tm1)
  v_t = rlax.batched_index(q_t, a_t)
  target_tm1 = rlax.lambda_returns(r_t, discount_t, v_t, lambda_)

  target_tm1 = jax.lax.select(stop_target_gradients,
                              jax.lax.stop_gradient(target_tm1), target_tm1)
  return target_tm1 - qa_tm1

@dataclasses.dataclass
class UsfaLossFn(basics.RecurrentLossFn):

  extract_cumulants: Callable = cumulants_from_env
  extract_task: Callable = lambda data: data.observation.observation['task']
  extract_task_dim: Callable = lambda x: x[:, :, 0]

  sf_coeff: float = 1.0
  q_coeff: float = 0.0
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
    online_sf = online_preds.sf
    online_z = online_preds.policy
    target_sf = target_preds.sf
    # pseudo rewards, [T/T+1, B, C]
    cumulants = self.extract_cumulants(
      data=data, online_preds=online_preds, online_state=online_state,
      target_preds=target_preds, target_state=target_state)
    cumulants = cumulants.astype(online_sf.dtype)

    # [T, B]
    episode_mask = make_episode_mask(data, include_final=True)
    online_sf = online_sf*episode_mask[:,:, None, None, None]
    target_sf = target_sf*episode_mask[:,:, None, None, None]


    # Get selector actions from online Q-values for double Q-learning.
    online_q =  (online_sf*online_z).sum(axis=CUMULANT_AXIS) # [T+1, B, N, A]
    selector_actions = jnp.argmax(online_q, axis=ACTION_AXIS) # [T+1, B, N]
    online_actions = data.action # [T, B]

    # Preprocess discounts & rewards.
    discounts = (data.discount * self.discount).astype(online_q.dtype) # [T, B]
    lambda_ = (data.discount * self.lambda_).astype(online_q.dtype) 

    # ======================================================
    # Prepare loss (via vmaps)
    # ======================================================
    def td_error_fn(
      online_sf_,
      online_actions_,
      target_sf_,
      selector_actions_,
      cumulants_,
      discounts_,
      lambda__,
    ):
      if self.loss_fn == 'qlearning':
        error_fn = functools.partial(
                rlax.transformed_n_step_q_learning,
                n=self.bootstrap_n)
        # vmap over cumulant dimension (C), return in dim=3
        error_fn = jax.vmap(
          error_fn,
          in_axes=(2, None, 2, None, 1, None), out_axes=1)
        return error_fn(
          online_sf_,  # [T, A, C]
          online_actions_,  # [T]         
          target_sf_,  # [T, A, C]
          selector_actions_,  # [T]      
          cumulants_,  # [T, C]      
          discounts_)  # [T]         

      elif self.loss_fn == 'qlambda':
        error_fn = jax.vmap(q_learning_lambda,
          in_axes=(2, None, 1, None, 2, None, None), out_axes=1)

        return error_fn(
          online_sf_,         # [T, A, C] (vmap 2)
          online_actions_,    # [T]       (vmap None)
          cumulants_,         # [T, C]    (vmap 1)
          discounts_,         # [T]       (vmap None)
          target_sf_,         # [T, A, C] (vmap 2)
          selector_actions_,  # [T]      
          lambda__,           # [T]       (vmap None)
        )
      else:
        raise NotImplementedError

    # vmap over batch dimension (B), return B in dim=1
    td_error_fn = jax.vmap(td_error_fn, in_axes=BATCH_AXIS, out_axes=BATCH_AXIS)

    # vmap over policy dimension (N), return N in dim=2
    td_error_fn = jax.vmap(
      td_error_fn,
      in_axes=(POLICY_AXIS, None, POLICY_AXIS, POLICY_AXIS, None, None, None),
      out_axes=POLICY_AXIS)

    # output = [0=T, 1=B, 2=N, 3=C]
    sf_td_error = td_error_fn(
      online_sf[:-1],  # [T, B, N, A, C] (vmap 2,1)
      online_actions[:-1],  # [T, B]          (vmap None,1)
      target_sf[1:],  # [T, B, N, A, C] (vmap 2,1)
      selector_actions[1:],  # [T, B, N]       (vmap 2,1)
      cumulants,  # [T-1, B, C]       (vmap None,1)
      discounts[:-1],  # [T, B]          (vmap None,1)
      lambda_[:-1],
      )

    #---------------------
    # Q-learning TD-error
    # needs to happen now because will be averaging over dim N
    #  below this.
    #---------------------
    task_td = self.extract_task_dim(sf_td_error)
    task_w = self.extract_task(data)[:-1]
    task_weights = task_w
    if self.indicator_weights:
      task_weights = (jnp.abs(task_w) < 1e-5).astype(task_w.dtype)  # float

    # [T, B, C]*[T, B, C] = [T, B]
    value_td_error = (task_td*task_weights).sum(-1)

    #---------------------
    # Compute average loss for SFs
    #---------------------
    # [T, B, N, C] --> [T, B]
    # mean over cumulants, mean over # of policies
    if self.sum_cumulants:
      sf_td_error = sf_td_error.sum(axis=3).mean(axis=2)
    else:
      sf_td_error = sf_td_error.mean(axis=3).mean(axis=2)

    if self.mask_loss:
      # [T, B]
      loss_mask = make_episode_mask(data, include_final=False)
      sf_loss = episode_mean(
        x=(0.5 * jnp.square(sf_td_error)),
        mask=loss_mask[:-1])
    else:
      sf_loss = (0.5 * jnp.square(sf_td_error)).mean(axis=(0,2,3))

    batch_loss = sf_loss*self.sf_coeff
    #---------------------
    # Compute average loss for Q-values
    #---------------------

    q_loss = jnp.zeros_like(batch_loss)
    if self.q_coeff > 0.0:
      if self.mask_loss:
        # [T, B]
        q_loss = episode_mean(
          x=(0.5 * jnp.square(value_td_error)),
          mask=loss_mask[:-1])
      else:
        q_loss = (0.5 * jnp.square(value_td_error)).mean(axis=(0,2,3))

      batch_loss = batch_loss + q_loss*self.q_coeff

    cumulant_reward = (cumulants*task_w).sum(-1)
    reward_error = (cumulant_reward - data.reward[:-1])

    if online_preds.action_mask is not None:
      mask = online_preds.action_mask[:,:,None]  # [T, B, 1, A]

      # [T, B, N, A]
      online_q = online_q*mask

      # [T, B, N, A, D]
      online_sf = online_sf*mask[:,:,:,:, None]

    metrics = {
      '0.total_loss': batch_loss,
      f'0.1.sf_loss_{self.loss_fn}': sf_loss,
      '0.1.q_loss': q_loss,
      '2.q_td_error': jnp.abs(value_td_error),
      '2.sf_td_error': jnp.abs(sf_td_error),
      '3.cumulants': cumulants,
      '3.sf_mean': online_sf,
      '3.sf_var': online_sf.var(axis=4),
      '3.q_mean': online_q,
      '4.reward_error': reward_error,
      }

    return sf_td_error, batch_loss, metrics # [T, B], [B]

class Observer(ActorObserver):
  def __init__(self,
               period=100,
               plot_success_only: bool = False,
               colors=None,
               prefix: str = 'SFsObserver'):
    super(Observer, self).__init__()
    self.period = period
    self.prefix = prefix
    self.successes = 0
    self.idx = -1
    self.logging = True
    self.plot_success_only = plot_success_only
    self._colors = colors or plt.rcParams['axes.prop_cycle']

  def wandb_log(self, d: dict):
    if wandb.run is not None:
      wandb.log(d)
    else:
      pass

  def observe_first(self, state: ActorState, timestep: dm_env.TimeStep) -> None:
    """Observes the initial state and initial time-step.

    Usually state will be all zeros and time-step will be output of reset."""
    self.idx += 1

    # epsiode just ended, flush metrics if you want
    if self.idx > 0:
      self.flush_metrics()

    # start collecting metrics again
    self.actor_states = [state]
    self.timesteps = [timestep]
    self.actions = []

  def observe_action(self, state: ActorState, action: jax.Array) -> None:
    """Observe state and action that are due to observation of time-step.

    Should be state after previous time-step along"""
    self.actor_states.append(state)
    self.actions.append(action)

  def observe_timestep(self, timestep: dm_env.TimeStep) -> None:
    """Observe next.

    Should be time-step after selecting action"""
    self.timesteps.append(timestep)

  def flush_metrics(self) -> Dict[str, float]:
    """Returns metrics collected for the current episode."""
    rewards = jnp.stack([t.reward for t in self.timesteps])[1:]
    total_reward = rewards.sum()
    if self.plot_success_only and not total_reward > 0:
      return

    self.successes += 1

    count = self.successes if self.plot_success_only else self.idx

    if not count % self.period == 0:
      return
    if not self.logging:
      return

    tasks = [t.observation.observation['task'] for t in self.timesteps]
    tasks = np.stack(tasks)
    task = tasks[0]

    # NOTES:
    # Actor states: T, starting at 0
    # timesteps: T, starting at 0
    # action: T-1, starting at 0
    ##################################
    # successor features
    ##################################
    # first prediction is empty (None)
    # [T, N, A, C]
    predictions = [s.predictions for s in self.actor_states[1:]]
    sfs = jnp.stack([p.sf for p in predictions])
    sfs = sfs[:, 0]  # [T-1, A, C]

    npreds = len(sfs)
    actions = jnp.stack(self.actions)[:npreds]

    index = jax.vmap(rlax.batched_index, in_axes=(2, None), out_axes=1)

    sfs = index(sfs, actions)  # [T-1, C]

    q_values = (sfs*tasks[:-1]).sum(-1)

    # ignore 0th (reset) time-step w/ 0 reward and last (terminal) time-step
    state_features = jnp.stack([t.observation.observation['state_features'] for t in self.timesteps])[1:]

    # Determine the number of plots needed based on the condition
    ndims = state_features.shape[1]
    active_dims = [j for j in range(ndims) if state_features[:, j].sum() > 0]
    if active_dims:
      n_plots = len(active_dims) + 1  # +1 for the rewards subplot

      # Calculate rows and columns for subplots
      cols = min(n_plots, 4)  # Maximum of 3 horizontally
      rows = np.ceil(n_plots / cols).astype(int)

      # Create a figure with dynamic subplots
      width = 3*cols
      height = 3*rows
      fig, axs = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)
      default_cycler = iter(self._colors)

      # Plot rewards in the first subplot
      axs[0, 0].plot(rewards, label='rewards', linestyle='--', color='grey')
      axs[0, 0].plot(q_values, label='q_values', color='grey')
      axs[0, 0].set_title("Reward Predictions")
      axs[0, 0].legend()

      # Initialize subplot index for state_features and sfs
      subplot_idx = 1  # Start from the second subplot
      for j in active_dims:
          # Calculate subplot position
          row, col = divmod(subplot_idx, cols)
          ax = axs[row, col]

          try:
              color = next(default_cycler)['color']
          except StopIteration:
              raise RuntimeError(f"unexpectedly had {len(active_dims)} dimensions active")

          # Plot state_features and sfs for each active dimension
          ax.plot(state_features[:, j], label=f'$\\phi - {j}$', linestyle='--', color=color)
          ax.plot(sfs[:, j], label=f'$\\psi - {j}$', color=color)
          ax.set_title(f"Dimension {j}")
          ax.legend()

          subplot_idx += 1  # Move to the next subplot

      # Hide any unused subplots if there are any
      for k in range(subplot_idx, rows * cols):
          row, col = divmod(k, cols)
          axs[row, col].axis('off')

      plt.tight_layout()
      self.wandb_log({f"{self.prefix}/sf-group-prediction": wandb.Image(fig)})
      plt.close(fig)
    else:
      if rewards.sum() > 0:
        raise RuntimeError("there is reward but no cumulants active. this must be a bug")

    # Close the plot
    # ##################################
    # # get images
    # ##################################
    # # [T, H, W, C]
    # frames = np.stack([t.observation.observation['image'] for t in self.timesteps])
    # # self.wandb_log({
    # #   f'{self.prefix}/episode-{task}': [wandb.Image(frame) for frame in frames]})
    # wandb.log({f'{self.prefix}/episode-{task}':
    #            wandb.Video(np.transpose(frames, (0, 3, 1, 2)), fps=.01)})

class USFAPreds(NamedTuple):
  q_values: jnp.ndarray  # q-value
  sf: jnp.ndarray # successor features
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
  1. Modular Successor Feature Approximators: https://arxiv.org/abs/2301.12305
  2. Categorical Successor Feature Approximators: https://arxiv.org/abs/2310.15940

  """
  def __init__(
      self,
      layers: int,
      num_actions: int,
      state_features_dim: int,
      activation: str = 'relu',
      compositional_policy: bool = False,
      policy_layers : Tuple[int]=(),
      name: Optional[str] = None,
      **kwargs):
    super(IndependentSfHead, self).__init__(name=name)

    self.sf_net_factory = lambda: hk.Sequential([
        hk.nets.MLP(tuple(layers), 
                    activation=getattr(jax.nn, activation),
                    activate_final=True,
                    with_bias=False),
        hk.Linear((num_actions), w_init=jnp.zeros, with_bias=False),
      ])

    self.policy_layers = policy_layers
    self.compositional_policy = compositional_policy


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

    linear = lambda x: hk.Linear(self.layers[0], with_bias=False)(x)
    concat = lambda a, b: jnp.concatenate((a, b))

    if self.compositional_policy:
      # [C, 1]
      policy = jnp.expand_dims(policy, axis=1)
      if self.policy_layers:
        # [[P], ..., [P]]
        policy = [self.policy_net_factory()(policy[idx]) for idx in range(ndims)]
        # [C, P]
        policy = jnp.stack(policy, axis=0)

      # Below, we make C copies of sf_input, break policy into C vectors of dimension 1,
      # and then concatenate each break policy unit to each sf_input copy.
      concat = jax.vmap(concat, in_axes=(None, 0), out_axes=0)
      sf_inputs = concat(sf_input, policy)  # [C, D+D]
      # now we get sf-estimates for each policy dimension
      # [[A], ..., [A]]
      sfs = [self.sf_net_factory()(sf_inputs[idx]) for idx in range(ndims)]
    else:
      policy = self.policy_net_factory()(policy)
      sf_input = concat(sf_input, policy)  # [C, D+D]
      sf_input = jax.nn.relu(linear(sf_input))
      sfs = [self.sf_net_factory()(sf_input) for _ in range(ndims)]

    # [A, C]
    sf = jnp.stack(sfs, axis=1)

    def dot(a, b): return jnp.sum(a*b).sum()

    # dot-product: A
    q_values = jax.vmap(
        dot, in_axes=(0, None), out_axes=0)(sf, task)

    assert q_values.shape[0] == self.num_actions, 'wrong shape'
    return sf, q_values

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

    sfs, q_values = jax.vmap(
      self.sf_net, in_axes=(None, 0, None), out_axes=0)(
        usfa_input,
        policies,
        task)

    # GPI
    # -----------------------
    # [N, A] --> [A]
    q_values = jnp.max(q_values, axis=-2)
    num_actions = q_values.shape[-1]

    # [N, D] --> [N, A, D]
    policies_repeated = jnp.repeat(policies[:, None], repeats=num_actions, axis=1)

    return USFAPreds(
      sf=sfs,       # [N, A, D_w]
      policy=policies_repeated,         # [N, A, D_w]
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
      SfNetCls = IndependentSfHead
    elif config.head == 'monolithic':
      SfNetCls = MonolithicSfHead
    else:
      raise NotImplementedError

    sf_net = SfNetCls(
      layers=config.sf_layers,
      state_features_dim=state_features_dim,
      num_actions=num_actions,
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
