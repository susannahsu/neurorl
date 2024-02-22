
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
# from projects.human_sf import usfa

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

  head: str = 'independent'
  sf_activation: str = 'relu'
  sf_mlp_type: str = 'hk'
  out_init_value: Optional[float] = 0.0

  # learning
  loss_fn : str = 'qlambda'
  sf_coeff: float = 1.0
  q_coeff: float = 1.0
  off_task_weight: float = .1
  sf_lambda: float = .9
  sum_cumulants: bool = True
  learning_support: str = 'train'

  weight_type: str = 'reg'
  combination: str = 'loss'

def cumulants_from_env(
    data, online_preds=None, online_state=None, target_preds=None, target_state=None):
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
class SFLossFn:

  weight_type: str = 'reg'
  combination: str = 'loss'

  weighted_coeff: float = 1.0
  unweighted_coeff: float = 1.0
  bootstrap_n: int = 5
  sum_cumulants: bool = True
  loss_fn : str = 'qlambda'

  def __call__(
    self,
    online_sf: jax.Array,
    online_actions: jax.Array,
    task_w: jax.Array,
    cumulants: jax.Array,
    discounts: jax.Array,
    target_sf: jax.Array,
    lambda_: jax.Array,
    loss_mask: Optional[jax.Array] = None,
    action_mask: Optional[jax.Array] = None,
    ):

    # Get selector actions from online Q-values for double Q-learning.
    online_q =  (online_sf*task_w).sum(axis=-1) # [T+1, A]
    selector_actions = jnp.argmax(online_q, axis=-1) # [T+1]
    # import ipdb; ipdb.set_trace()
    if self.loss_fn == 'qlearning':
      error_fn = functools.partial(
        rlax.transformed_n_step_q_learning,
        n=self.bootstrap_n)
      # vmap over cumulant dimension (C), return in dim=3
      error_fn = jax.vmap(
        error_fn,
        in_axes=(2, None, 2, None, 1, None), out_axes=1)
      td_error = error_fn(
        online_sf,  # [T, A, C]
        online_actions,  # [T]         
        target_sf,  # [T, A, C]
        selector_actions,  # [T]      
        cumulants,  # [T, C]      
        discounts)  # [T]         

    elif self.loss_fn == 'qlambda':
      error_fn = jax.vmap(q_learning_lambda,
        in_axes=(2, None, 1, None, 2, None, None), out_axes=1)

      td_error = error_fn(
        online_sf,         # [T, A, C] (vmap 2)
        online_actions,    # [T]       (vmap None)
        cumulants,         # [T, C]    (vmap 1)
        discounts,         # [T]       (vmap None)
        target_sf,         # [T, A, C] (vmap 2)
        selector_actions,  # [T]      
        lambda_,           # [T]       (vmap None)
      )
    else:
      raise NotImplementedError

    loss_fn = lambda x: (0.5 * jnp.square(x))
    ####################
    # what will be the loss weights?
    ####################
    # [T, A, D], shared across actions
    td_weights = task_w[:, 0]
    if self.weight_type == 'mag':
      td_weights = jnp.abs(td_weights)
    elif self.weight_type == 'reg':
      pass
    elif self.weight_type == 'indicator':
      td_weights = (jnp.abs(td_weights) < 1e-5)
    else:
      raise NotImplementedError

    td_weights = td_weights.astype(online_sf.dtype)  # float

    ####################
    # how will we combine them into the loss so we're optimizing towards task
    ####################
    # [T, C] --> [T]
    T, C = td_error.shape
    task_weighted_td_error = (td_error*td_weights).sum(-1)
    if self.combination == 'td':
      # in this setting, tds were weighted and apply loss there
      # [T]
      task_weighted_loss = loss_fn(task_weighted_td_error)
    elif self.combination == 'loss':
      # in this setting, weight the output of the loss
      # [T, C]
      td_loss = loss_fn(td_error)
      # [T]
      task_weighted_loss = (td_loss*td_weights).sum(-1)
    else:
      raise NotImplementedError
    assert task_weighted_loss.shape == (T,)


    ####################
    # computing regular loss
    ####################
    # [T]
    if self.sum_cumulants:
      unweighted_td_error = td_error.sum(axis=-1)
    else:
      unweighted_td_error = td_error.mean(axis=-1)
    unweighted_loss = loss_fn(unweighted_td_error)

    ####################
    # combine losses
    ####################
    if loss_mask is not None:
      assert loss_mask.shape[0] == task_weighted_loss.shape[0]
      assert loss_mask.shape[0] == unweighted_loss.shape[0]
      task_weighted_loss = episode_mean(task_weighted_loss, loss_mask)
      unweighted_loss = episode_mean(unweighted_loss, loss_mask)

    batch_loss = (task_weighted_loss * self.weighted_coeff + 
                  unweighted_loss * self.unweighted_coeff)
    batch_td_error = (task_weighted_td_error * self.weighted_coeff + 
                      unweighted_td_error * self.unweighted_coeff)

    ####################
    # metrics
    ####################

    if action_mask is not None:
      # [T,A]*[T,A]
      online_q = online_q*action_mask

      # [T,A,C]*[T,A,1]
      online_sf = online_sf*action_mask[:,:, None]

    metrics = {
      '0.total_loss': batch_loss,
      f'0.1.unweighted_loss': unweighted_loss,
      '0.1.task_weighted_loss': task_weighted_loss,
      '2.td_error': jnp.abs(batch_td_error),
      '3.cumulants': cumulants,
      '3.sf_mean': online_sf,
      '3.sf_var': online_sf.var(axis=-1),
      '3.q_mean': online_q,
      # '4.reward_error': reward_error,
      }

    # [T], []
    return batch_td_error, batch_loss, metrics



@dataclasses.dataclass
class MultitaskUsfaLossFn(basics.RecurrentLossFn):

  extract_cumulants: Callable = cumulants_from_env
  # extract_task: Callable = lambda data: data.observation.observation['task']
  # extract_task_dim: Callable = lambda x: x[:, :, 0]

  sf_coeff: float = 1.0
  q_coeff: float = 1.0
  loss_fn : str = 'qlambda'
  lambda_: float  = .9
  sum_cumulants: bool = True
  weight_type: str = 'reg'
  combination: str = 'td'
  off_task_weight: float = .1

  def error(self, data, online_preds, online_state, target_preds, target_state, **kwargs):
    # ======================================================
    # Prepare Data
    # ======================================================
    #---------------
    # SFs
    #---------------
    # all are [T+1, B, N, A, C]
    # N = num policies, A = actions, C = cumulant dim
    online_sf = online_preds.sf  # [T+1, B, N, A, C]
    online_z = online_preds.policy  # [T+1, B, N, A, C]
    target_sf = target_preds.sf  # [T+1, B, N, A, C]

    # [T, B]
    # episode_mask = make_episode_mask(data, include_final=True)
    loss_mask = make_episode_mask(data, include_final=False)
    online_sf = online_sf*loss_mask[:,:, None, None, None]
    target_sf = target_sf*loss_mask[:,:, None, None, None]

    #---------------
    # cumulants
    #---------------
    # pseudo rewards, [T/T+1, B, C]
    cumulants = self.extract_cumulants(data=data)
    cumulants = cumulants.astype(online_sf.dtype)

    #---------------
    # discounts
    #---------------
    discounts = (data.discount * self.discount).astype(online_sf.dtype) # [T, B]
    lambda_ = (data.discount * self.lambda_).astype(online_sf.dtype) 


    sf_loss_fn = SFLossFn(
      weight_type=self.weight_type,
      combination=self.combination,
      weighted_coeff=self.q_coeff,
      unweighted_coeff=self.sf_coeff,
      sum_cumulants=self.sum_cumulants,
      loss_fn=self.loss_fn,
    )

    # NOTE: outputs will lose a dimension, so lowering out_axes by 1
    # vmap over batch dimension (B), return B in dim=0
    sf_loss_fn = jax.vmap(
      sf_loss_fn, in_axes=BATCH_AXIS, out_axes=BATCH_AXIS-1)

    # vmap over policy dimension (N), return N in dim=2
    in_axes = (
      POLICY_AXIS,
      None,
      POLICY_AXIS,
      None,
      None,
      POLICY_AXIS,
      None,
      None)
    args = (
      online_sf[:-1], 
      data.action[:-1],
      online_z[:-1],
      cumulants,
      discounts[:-1],
      target_sf[1:],
      lambda_[:-1],
      loss_mask[:-1],
    )

    if online_preds.action_mask is not None:
      action_mask = online_preds.action_mask.astype(online_sf.dtype)
      args = args + (action_mask[:-1],)
      in_axes = in_axes + (None, )

    sf_loss_fn = jax.vmap(
      sf_loss_fn,
      in_axes=in_axes,
      out_axes=POLICY_AXIS-1)

    # [B, N, T], # [B, N]
    td_error, batch_loss, metrics = sf_loss_fn(*args)

    npolicies = td_error.shape[1]
    loss_weights = (1.0,)
    if npolicies > 1: 
      loss_weights += (self.off_task_weight,)*(npolicies-1)

    # [1, N]
    loss_weights = jnp.array(loss_weights)[None]

    # average over policies
    td_error = (td_error*loss_weights[:,:,None]).mean(1)

    batch_loss = (batch_loss*loss_weights).mean(1)

    # parent call expects td_error to be [T, B]
    # [B, T] --> [T, B]
    td_error = jnp.transpose(td_error, axes=(1, 0))

    return td_error, batch_loss, metrics


###################################
# Actor Observer
###################################

def non_zero_elements_to_string(arr):
  """
  Generates a string that lists the indices and values of non-zero elements in a NumPy array.
  
  Parameters:
  - arr: A NumPy array.
  
  Returns:
  - A string in the format "{index1}={value1}, {index2}={value2},..." for non-zero elements.
  """
  # Find indices where the array is not equal to zero
  non_zero_indices = np.nonzero(arr)[0]
  # Extract the non-zero values based on the indices
  non_zero_values = arr[non_zero_indices]
  # Format the indices and their corresponding values into the requested string format
  result = ",".join(f"{index}:{value:.2g}" for index, value in zip(non_zero_indices, non_zero_values))
  return f"[{result}]"

def plot_sfgpi(
    sfs: np.array, actions: np.array, train_tasks: np.array, frames: np.array, max_cols: int = 10, title:str = ''):
  max_len = min(sfs.shape[0], 100)
  sfs = sfs[:max_len]
  train_tasks = train_tasks[:max_len]
  frames = frames[:max_len]

  max_sf = max(sfs.max(), .1)

  T, N, A, C = sfs.shape  # Time steps, N, Actions, Channels for sfs
  T2, N2, C2 = train_tasks.shape  # Time steps, N, Channels for train_tasks

  # Validate dimensions
  if C != C2 or N != N2 or T != T2:
      raise ValueError("Dimensions of sfs and train_tasks do not match as expected.")

  # Compute train_q_values with corrected dimension alignment
  train_q_values = (sfs * train_tasks[:, :, None]).sum(-1)  # [T, N, A]
  # actions_from_q = train_q_values.max(1).argmax()
  # assert (actions_from_q == actions).all(), "computed incorrectly, mismatch"
  max_q = max(train_q_values.max(), .1)

  # Determine the layout
  cols = min(T, max_cols)
  # Adjust total rows calculation to include an additional N rows for the heatmaps
  total_rows = (T // max_cols) * (2 + N)  # 2 rows for images and bar plots, plus N rows for heatmaps
  if T % max_cols > 0:  # Add additional rows if there's a remainder
      total_rows += (2 + N)

  # Prepare the matplotlib figure
  unit = 3
  fig, axs = plt.subplots(total_rows, cols, figsize=(cols*unit, total_rows*unit), squeeze=False)
  fig.suptitle(title, fontsize=16, y=1.03)

  # Iterate over time steps and plot
  total_plots = total_rows // (2 + N) * cols  # Recalculate total plots based on new row structure
  for t in range(total_plots):
    # Calculate base row index for current time step; adjusted for 2 rows per image/bar plot plus N rows for heatmaps
    base_row = (t // max_cols) * (2 + N)
    col = t % max_cols  # Column wraps every max_cols

    if t < T:
      axs[base_row, col].set_title(f"t={t+1}")

      # Plot frames
      axs[base_row, col].imshow(frames[t])
      axs[base_row, col].axis('off')  # Turn off axis for images

      # Plot bar plots
      # [N, A] --> [N]
      max_q_values = train_q_values[t].max(axis=-1)
      max_index = max_q_values.argmax()  # best N

      colors = ['red' if i == max_index else 'blue' for i in range(N)]
      axs[base_row+1, col].bar(range(N), max_q_values, color=colors)
      task_labels = [non_zero_elements_to_string(i) for i in train_tasks[t]]
      axs[base_row+1, col].set_xticks(range(N))  # Set tick positions
      axs[base_row+1, col].set_xticklabels(task_labels, rotation=0)
      axs[base_row+1, col].set_ylim(0, max_q * 1.1)  # Set y-axis limit to 1.
      axs[base_row+1, col].set_title(f"Chosen={max_index+1}, a ={actions[t]}")  # Set y-axis limit to 1.

      # Plot heatmaps for each N
      for n in range(N):
          non_zero_indices = np.nonzero(train_tasks[t,n])[0]
          colors = ['black' if i in non_zero_indices else 'skyblue' for i in range(C)]
          # Identify the action with the highest Q-value for this N at time t
          action_with_highest_q = train_q_values[t, n].argmax()
          
          # Extract SFs values for this action
          sf_values_for_highest_q = sfs[t, n, action_with_highest_q, :]
          
          # Plot barplot of SFs for the highest Q-value action
          axs[base_row+2+n, col].bar(range(C), sf_values_for_highest_q, color=colors)
          axs[base_row+2+n, col].set_title(f"policy {n+1}, a = {action_with_highest_q}")

          axs[base_row+2+n, col].set_ylim(0, max_sf * 1.1)  # Set y-axis limit to 1.
          axs[base_row+2+n, col].axis('on')  # Optionally, turn on the axis if needed

    else:
        # Remove unused axes
        for r_offset in range(2 + N):
            try:
                fig.delaxes(axs[base_row + r_offset, col])
            except:
                break

  axs[0, 0].set_title(title)
  plt.tight_layout()
  return fig

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
    self.failures = 0
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
    is_success = total_reward > 1.
    if total_reward > 1:
      self.successes += 1
    elif total_reward > 1e-3:
      self.failures += 1
    else:
      return

    success_period = self.successes % self.period == 0
    failure_period = self.failures % self.period == 0

    if not (success_period or failure_period):
      return

    # [T, C]
    tasks = [t.observation.observation['task'] for t in self.timesteps]
    tasks = np.stack(tasks)

    # [T, N, C]
    train_tasks = [t.observation.observation['train_tasks'] for t in self.timesteps]
    train_tasks = np.stack(train_tasks)
    
    sfs = [s.predictions.sf for s in self.actor_states[1:]]
    sfs = np.stack(sfs)

    frames = np.stack([t.observation.observation['image'] for t in self.timesteps])


    # e.g. "Success 4: 0=1, 4=.5, 5=.5"
    task_str = non_zero_elements_to_string(tasks[0])
    title_prefix = f'success {self.successes}' if is_success else f'failure {self.failures}'
    title = f"{title_prefix}. Task: {task_str}"


    fig = plot_sfgpi(
      sfs=sfs,
      train_tasks=train_tasks[:-1],
      frames=frames,
      title=title
      )

    wandb_suffix = 'success' if is_success else 'failure'
    self.wandb_log({f"{self.prefix}/sfgpi-{wandb_suffix}": wandb.Image(fig)})
    plt.close(fig)

    ##################################
    # successor features
    ##################################
    npreds = len(sfs)
    actions = jnp.stack(self.actions)[:npreds]

    # sfs: [T, N, C]
    index = jax.vmap(rlax.batched_index, in_axes=(2, None), out_axes=1)
    index = jax.vmap(index, in_axes=(1, None), out_axes=1)


    sfs = index(sfs, actions)  # [T-1, N, C]

    q_values = (sfs*tasks[:-1, None]).sum(-1).max(-1)  # gpi

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

      # fig.suptitle(title, fontsize=16, y=1.03)

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

          default_cycler = iter(self._colors)
          # Plot state_features and sfs for each active dimension
          ax.plot(state_features[:, j], label=f'$\\phi_{j}$', linestyle='--')
          for n in range(sfs.shape[1]):
            # try:
            #     color = next(default_cycler)['color']
            # except StopIteration:
            #     raise RuntimeError(f"too many policies?")
            ax.plot(sfs[:, n, j], label=f'$\\pi_{n}, \\psi_{j}$')
          ax.set_title(f"Dimension {j}")
          ax.legend()

          subplot_idx += 1  # Move to the next subplot

      # Hide any unused subplots if there are any
      for k in range(subplot_idx, rows * cols):
          row, col = divmod(k, cols)
          axs[row, col].axis('off')

      plt.tight_layout()
      axs[0, 0].set_title(title)
      self.wandb_log({f"{self.prefix}/sf-predictions-{wandb_suffix}": wandb.Image(fig)})
      plt.close(fig)
    # else:
    #   if rewards.sum() > 0:
    #     raise RuntimeError("there is reward but no cumulants active. this must be a bug")

    # Close the plot
    # ##################################
    # # get images
    # ##################################
    #
    # import ipdb; ipdb.set_trace()
    # model_predictions = [s.predictions.model_predictions.predictions for s in self.actor_states[1:]]

    # # [T, sim, tasks, actions, cumulants]
    # sfs = jnp.stack([p.sf for p in model_predictions])
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

    linear_dim = self.layers[0] if self.layers else sf_input.shape[-1]
    linear = lambda x: hk.Linear(linear_dim, with_bias=False)(x)
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
               learning_support: str = 'train_tasks',
               name: str = 'usfa_arch'):
    super().__init__(name=name)
    self._torso = torso
    self._memory = memory
    self._head = head
    self._learning_support = learning_support

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> hk.LSTMState:
    return self._memory.initial_state(batch_size)

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

    if self._learning_support == 'train_tasks':
      # [T, B, N, D]
      support = jnp.concatenate(
        (jnp.expand_dims(inputs.observation['task'], axis=-2),
        inputs.observation['train_tasks']),
        axis=2
        )
    elif self._learning_support == 'eval':
      # [T, B, 1, D]
      support = jnp.expand_dims(inputs.observation['task'], axis=-2)
    else:
      raise NotImplementedError(self._learning_support)

    # treat T,B like this don't exist with vmap
    predictions = jax.vmap(jax.vmap(self._head.evaluate))(
        inputs.observation['task'],  # [T, B]
        core_outputs,                # [T, B, D]
        support,
      )
    return predictions, new_states

def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config) -> networks_lib.UnrollableNetwork:
  """Builds default USFA networks for Minigrid games."""

  num_actions = env_spec.actions.num_values
  state_features_dim = env_spec.observations.observation['state_features'].shape[0]

  def make_core_module() -> UsfaArch:

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
      torso=networks.OarTorso(
        num_actions=num_actions,
        vision_torso=networks.BabyAIVisionTorso(
          conv_dim=config.final_conv_dim,
          out_dim=config.conv_flat_dim),
        output_fn=networks.TorsoOutput,
      ),
      memory=hk.LSTM(config.state_dim),
      head=usfa_head,
      learning_support=config.learning_support,
      )

  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)
