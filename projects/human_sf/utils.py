from typing import Optional, Callable, NamedTuple, Dict
from acme.wrappers import observation_action_reward
import haiku as hk

import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import rlax
import wandb

from td_agents.basics import ActorObserver, ActorState
from library.utils import LevelAvgObserver

Array = jax.Array


def add_batch(nest, batch_size: Optional[int]):
  """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
  broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
  return jax.tree_util.tree_map(broadcast, nest)

def process_inputs(
    inputs: observation_action_reward.OAR,
    vision_fn: hk.Module,
    task_encoder: Optional[hk.Module] = None,
    ):
  # # [A, A] array
  # actions = jax.nn.one_hot(
  #     inputs.observation['actions'],
  #     num_classes=num_primitive_actions)

  # convert everything to floats
  inputs = jax.tree_map(lambda x: x.astype(jnp.float32), inputs)

  #----------------------------
  # process actions
  #----------------------------
  # 1 -> A+1: primitive actions
  # rest: object 
  # each are [N, D] where N differs for action and object embeddings
  def mlp(x):
    x = hk.Linear(256, w_init=hk.initializers.TruncatedNormal())(x)
    x = hk.Linear(256)(jax.nn.relu(x))
    return x
  # action_mlp = hk.to_module(mlp)('action_mlp')
  object_mlp = hk.to_module(mlp)('object_mlp')

  # actions = action_mlp(actions)
  objects = object_mlp(inputs.observation['objects'])

  #----------------------------
  # compute selected action
  #----------------------------
  # prev_objects = object_mlp(prev_objects)
  # all_actions = jnp.concatenate((actions, prev_objects), axis=-2)
  # chosen_action = all_actions[prev_action]

  #----------------------------
  # image + task + reward
  #----------------------------
  image = inputs.observation['image']/255.0
  image = vision_fn(image)  # [D+A+1]

  # task
  task = None
  if task_encoder:
    task = task_encoder(inputs.observation)

  # reward = jnp.tanh(inputs.reward)
  reward = jnp.expand_dims(inputs.reward, axis=-1)

  objects_mask = inputs.observation['objects_mask']
  return image, task, reward, objects, objects_mask

class TaskAwareLSTMState(NamedTuple):
  hidden: jax.Array
  cell: jax.Array
  task: jax.Array

def add_task_to_lstm_state(state, task):
  return TaskAwareLSTMState(
    hidden=state.hidden,
    cell=state.cell,
    task=task,
  )

class TaskAwareRecurrentFn(hk.RNNCore):
  """Helper RNN which adds task to state and, optionally, hidden output. 
  
  It's useful to couple the task g and the state s_t output for functions that make
    predictions at each time-step. For example:
      - value predictions: V(s_t, g)
      - policy: pi(s_t, g)
    If you use this class with hk.static_unroll, then the hidden output will have s_t 
      and g of the same dimension.
      i.e. ((s_1, g), (s_2, g), ..., (s_k, g))
  """
  def __init__(self,
               core: hk.LSTMState,
               task_dim: Optional[int] = None,
               couple_state_task: bool = True,
               get_task: Callable[
                   [Array, Array], Array] = lambda inputs, state: state.task,
               prep_input: Callable[
                  [Array], Array] = lambda x: x,
               prep_state: Callable[[Array], Array] = lambda x: x,
               name: Optional[str] = None
               ):
    super().__init__(name=name)
    self._core = core
    self._task_dim = task_dim
    self._get_task = get_task
    self._couple_state_task = couple_state_task
    self._prep_input = prep_input
    self._prep_state = prep_state

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> TaskAwareLSTMState:
    if not self._couple_state_task:
      return self._core.initial_state(batch_size)

    if self._task_dim is None:
      raise RuntimeError("Don't expect to initialize state")

    state = self._core.initial_state(None)
    task = jnp.zeros(self._task_dim, dtype=jnp.float32)
    state = add_task_to_lstm_state(state, task)

    if batch_size:
      state = add_batch(state, batch_size)
    return state

  def __call__(self,
               inputs: Array,
               prev_state: TaskAwareLSTMState,
               task: Optional[jax.Array] = None):
    prepped_input = self._prep_input(inputs)
    prepped_state = self._prep_state(prev_state)

    _, state = self._core(prepped_input, prepped_state)

    if task is None:
      task = self._get_task(inputs, prev_state)
    if self._couple_state_task:
      state = add_task_to_lstm_state(state, task)

    return state, state

def plot_frames_dynamically(task_name, frames, rewards, actions_taken, H, W):
  """
  Dynamically plots frames in multiple figures based on the specified grid layout (H, W).

  :param frames: 4D numpy array of shape (T, H, W, C) containing the frames to plot.
  :param H: Number of rows in each plot's grid.
  :param W: Number of columns in each plot's grid.
  """
  T = frames.shape[0]  # Total number of frames
  frames_per_plot = H * W  # Calculate the number of frames per plot based on the grid layout
  num_plots = T // frames_per_plot + int(T % frames_per_plot != 0)  # Number of plots needed

  width=3
  for plot_index in range(num_plots):
      fig, axs = plt.subplots(H, W, figsize=(W*width, H*width))
      # fig.suptitle(f'{task_name}: plot {plot_index + 1}')
      # Flatten the axes array for easy iteration if it's multidimensional
      if H * W > 1:
          axs = axs.ravel()
      for i in range(frames_per_plot):
          frame_index = plot_index * frames_per_plot + i
          if frame_index < T:
              if H * W == 1:  # If there's only one subplot, don't try to index axs
                  ax = axs
              else:
                  ax = axs[i]
              ax.imshow(frames[frame_index])
              ax.axis('off')  # Hide the axis
              if frame_index < len(actions_taken):
                if frame_index == 0:
                  ax.set_title(task_name)
                else:
                  ax.set_title(f"{actions_taken[frame_index]}, r={rewards[frame_index]}")
          else:  # Hide unused subplots
              if H * W > 1:  # Only if there are multiple subplots
                  axs[i].axis('off')

      plt.tight_layout()
      yield fig

class LevelEpisodeObserver(LevelAvgObserver):
  """Metric: Average return over many episodes"""

  def __init__(self, *args, get_action_names = None, **kwargs):
    super(LevelEpisodeObserver, self).__init__(*args, **kwargs)
    if get_action_names is None:
      get_action_names = lambda env: None

    self.get_action_names = get_action_names

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Observes the initial state __after__ reset."""
    self.idx += 1
    self.task_name = self.current_task(env)
    self.action_names = self.get_action_names(env)
    self.timesteps = [timestep]
    self.actions = []

  def observe(self,
              env: dm_env.Environment,
              timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Records one environment step."""
    self.timesteps.append(timestep)
    self.actions.append(action)


  def wandb_log(self, d: dict):
    if wandb.run is not None:
      wandb.log(d)
    else:
      pass
      # self.reset = np.inf

  def get_metrics(self):
    """Returns metrics collected for the current episode."""
    result = {}

    if self.idx % self.reset != 0:
      return result

    # [T, H, W, C]
    frames = np.stack([t.observation.observation['image'] for t in self.timesteps])
    rewards = np.stack([t.reward for t in self.timesteps[1:]])
    actions_taken = [self.action_names[int(a)] for a in self.actions]
    for fig in plot_frames_dynamically(self.task_name, frames, rewards, actions_taken, H=10, W=10):
       self.wandb_log({f"{self.prefix}/episode": wandb.Image(fig)})
       plt.close(fig)

    return result


###################################
# Actor Observer
###################################
def sorted_nonzero_indices(arr: np.array):
  # Step 1: Identify non-zero indices
  non_zero_indices = np.nonzero(arr)[0]

  # Step 2: Sort the non-zero indices by their corresponding values
  # Extract non-zero values using the indices
  non_zero_values = arr[non_zero_indices]

  # Get the sorted order of indices based on values
  sorted_order = np.argsort(non_zero_values)

  # Apply sorted order to non-zero indices
  sorted_non_zero_indices = non_zero_indices[sorted_order]
  return sorted_non_zero_indices


def non_zero_elements_to_string(arr: np.array):
  """
  Generates a string that lists the indices and values of non-zero elements in a NumPy array.
  
  Parameters:
  - arr: A NumPy array.
  
  Returns:
  - A string in the format "{index1}={value1}, {index2}={value2},..." for non-zero elements.
  """
  # Find indices where the array is not equal to zero
  non_zero_indices = sorted_nonzero_indices(arr)
  # Extract the non-zero values based on the indices
  non_zero_values = arr[non_zero_indices]
  # Format the indices and their corresponding values into the requested string format
  result = ",".join(f"{index}:{value:.2g}" for index, value in zip(non_zero_indices, non_zero_values))
  return f"[{result}]"

def plot_sfgpi(
    sfs: np.array,
    actions: np.array,
    chosen_q_values: np.array,
    train_q_values: np.array,
    train_tasks: np.array,
    tasks: np.array,
    frames: np.array,
    max_cols: int = 10, title:str = ''):
  max_len = min(sfs.shape[0], 100)
  sfs = sfs[:max_len]
  train_tasks = train_tasks[:max_len]
  frames = frames[:max_len]

  max_sf = max(sfs.max(), .1)

  ##########################
  # Setting up plots
  ##########################
  T, N, A, C = sfs.shape  # Time steps, N, Actions, Channels for sfs
  T2, N2, C2 = train_tasks.shape  # Time steps, N, Channels for train_tasks

  # Validate dimensions
  if C != C2 or N != N2 or T != T2:
      raise ValueError("Dimensions of sfs and train_tasks do not match as expected.")

  # max q-value
  max_q = max(train_q_values.max(), .1)

  # Determine the layout
  cols = min(T, max_cols)
  # Adjust total rows calculation to include an additional N rows for the heatmaps
  total_rows = (T // max_cols) * (3 + N)  # Adjusted for an additional row for the heatmap
  if T % max_cols > 0:
      total_rows += (3 + N)

  # Prepare the matplotlib figure
  unit = 3
  fig, axs = plt.subplots(total_rows, cols, figsize=(cols*unit, total_rows*unit), squeeze=False)
  fig.suptitle(title, fontsize=16, y=1.03)

  # Iterate over time steps and plot
  total_plots = total_rows // (2 + N) * cols  # Recalculate total plots based on new row structure

  ##########################
  # Iterating over time dimensions
  ##########################
  for t in range(total_plots):
    # Calculate base row index for current time step; adjusted for 2 rows per image/bar plot plus N rows for heatmaps
    base_row = (t // max_cols) * (3 + N)  # Adjusted for an additional row for the heatmap
    col = t % max_cols  # Column wraps every max_cols

    if t < T:

      #------------------
      # Plot image
      #------------------
      axs[base_row, col].set_title(f"{title}\nt={t+1}")
      axs[base_row, col].imshow(frames[t])
      axs[base_row, col].axis('off')  # Turn off axis for images

      #------------------
      # Plot maximum Q-values for each policy
      #------------------
      max_q_values = train_q_values[t].max(axis=-1)
      max_index = max_q_values.argmax()  # best N


      colors = ['red' if i == max_index else 'blue' for i in range(N)]
      axs[base_row+1, col].bar(range(N), max_q_values, color=colors)

      task_labels = [non_zero_elements_to_string(i) for i in train_tasks[t]]
      axs[base_row+1, col].set_xticks(range(N))  # Set tick positions
      axs[base_row+1, col].set_xticklabels(task_labels, rotation=0, fontsize=8)
      axs[base_row+1, col].set_ylim(0, max_q * 1.1)  # Set y-axis limit to 1.
      axs[base_row+1, col].set_title(f"Chosen={max_index+1}, a ={actions[t]}")  # Set y-axis limit to 1.

      #------------------
      # Plot heatmap for train_q_values
      #------------------
      train_q_values_T = train_q_values[t].transpose()
      axs[base_row+2, col].imshow(train_q_values_T, cmap='hot', interpolation='nearest')
      for i in range(train_q_values_T.shape[0]):
          for j in range(train_q_values_T.shape[1]):
              axs[base_row+2, col].text(j, i, f"{train_q_values_T[i,j]:.2g}", ha="center", va="center", color="w", fontsize=6)
      axs[base_row+2, col].set_title("Heatmap")

      #------------------
      # Plot heatmaps for each N
      #------------------
      non_zero_indices = np.nonzero(tasks[t])[0]
      colors = ['black' if i in non_zero_indices else 'skyblue' for i in range(C)]
      for n in range(N):
          # Identify the action with the highest Q-value for this N at time t
          action_with_highest_q = train_q_values[t, n].argmax()

          # Extract SFs values for this action
          sf_values_for_highest_q = sfs[t, n, action_with_highest_q, :]


          # Plot barplot of SFs for the highest Q-value action
          axs[base_row+3+n, col].bar(range(C), sf_values_for_highest_q, color=colors)
          axs[base_row+3+n, col].set_title(f"policy {n+1}, a = {action_with_highest_q}")
          axs[base_row+3+n, col].set_ylim(0, max_sf * 1.1)  # Set y-axis limit to 1.
          axs[base_row+3+n, col].axis('on')  # Optionally, turn on the axis if needed

          # Annotate bars for non-zero indices
          for i in non_zero_indices:
              height = sf_values_for_highest_q[i]
              axs[base_row+3+n, col].text(i, height, f'{tasks[t][i]:.2f}', ha='center', va='bottom')

    else:
        # Remove unused axes
        for r_offset in range(3 + N):
            try:
                fig.delaxes(axs[base_row + r_offset, col])
            except:
                break

  plt.tight_layout()
  return fig

class SFObserver(ActorObserver):

  def __init__(
      self,
      period=100,
      plot_success_only: bool = False,
      colors=None,
      prefix: str = 'SFsObserver'):
    super(SFObserver, self).__init__()
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

    if self.period == 0: return
    elif self.period == 1: pass
    else:
      if total_reward > 1:
        self.successes += 1
      elif total_reward > 1e-5:
        self.failures += 1
      else:
        return

      success_period = self.successes % self.period == 0
      failure_period = self.failures % self.period == 0

      if self.successes == 1 or self.failures == 1:
         pass
      elif not (success_period or failure_period):
        return

    def get_from_timesteps(key):
      x = [t.observation.observation[key] for t in self.timesteps]
      return np.stack(x)

    def get_from_predictions(key):
      x = [getattr(s.predictions, key) for s in self.actor_states[1:]]
      return np.stack(x)

    tasks = get_from_timesteps('task')  # [T, C]
    train_tasks = get_from_timesteps('train_tasks')  # [T, N, C]
    sfs = get_from_predictions('sf')  # [T, N, A, C]


    # [T, N, A]
    all_q_values = get_from_predictions('all_q_values')
    npreds = all_q_values.shape[0]
    actions = jnp.stack(self.actions)[:npreds]

    # [T, A]
    q_values = get_from_predictions('q_values')
    q_values = rlax.batched_index(q_values, actions)  # [T]

    frames = np.stack([t.observation.observation['image'] for t in self.timesteps])

    # e.g. "Success 4: 0=1, 4=.5, 5=.5"
    is_success = total_reward > 1.
    task_str = non_zero_elements_to_string(tasks[0])
    title_prefix = f'{self.successes}' if is_success else f'{self.failures}'
    title = f"{title_prefix}. {task_str}"

    fig = plot_sfgpi(
      sfs=sfs,
      actions=actions,
      chosen_q_values=q_values,
      train_q_values=all_q_values,
      train_tasks=train_tasks[:-1],
      tasks=tasks[:-1],
      frames=frames,
      title=title
      )

    wandb_suffix = 'success' if is_success else 'failure'
    self.wandb_log({f"{self.prefix}/sfgpi-{wandb_suffix}": wandb.Image(fig)})
    plt.close(fig)

    ##################################
    # line bars
    ##################################
    npreds = len(sfs)
    # sfs: [T, N, C]
    index = jax.vmap(rlax.batched_index, in_axes=(2, None), out_axes=1)
    index = jax.vmap(index, in_axes=(1, None), out_axes=1)
    sfs = index(sfs, actions)  # [T-1, N, C]


    # ignore 0th (reset) time-step w/ 0 reward and last (terminal) time-step
    state_features = jnp.stack([t.observation.observation['state_features'] for t in self.timesteps])[1:]

    # Determine the number of plots needed based on the condition
    # ndims = state_features.shape[1]
    # active_dims = [j for j in range(ndims) if state_features[:, j].sum() > 0]

    active_dims = sorted_nonzero_indices(tasks[0])

    if len(active_dims):
      cols = len(active_dims) + 1  # +1 for the rewards subplot

      # Calculate rows and columns for subplots
      # cols = min(n_plots, 4)  # Maximum of 3 horizontally
      rows = 2

      # Create a figure with dynamic subplots
      width = 3*cols
      height = 3*rows
      fig, axs = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)


      # Plot rewards in the first subplot
      axs[0, 0].set_title(title)
      axs[0, 0].plot(rewards, label='rewards', linestyle='--', color='grey')
      axs[0, 0].plot(q_values, label='q_values', color='grey')
      axs[0, 0].legend()
      assert sfs.shape[1] == all_q_values.shape[1], f'num policies do not match. shape={str(all_q_values.shape)}'

      for n in range(all_q_values.shape[1]):
        q_values_n = rlax.batched_index(all_q_values[:, n], actions)
        axs[1, 0].plot(q_values_n, label=f'$\\pi_{n}$ q_values')
      axs[1, 0].legend()
      axs[1, 0].set_title(f"total reward: {total_reward:.2g}")

      # Initialize subplot index for state_features and sfs
      for j in active_dims:
          # Calculate subplot position
          # row, col = divmod(subplot_idx, cols)
          col = j+1
          # default_cycler = iter(self._colors)
          # Plot state_features and sfs for each active dimension
          axs[0, col].plot(state_features[:, j], label=f'$\\phi_{j}$', linestyle='--', color='grey')
          axs[0, col].set_title(f"Dimension {j}")
          axs[0, col].legend()

          for n in range(sfs.shape[1]):
            # try:
            #     color = next(default_cycler)['color']
            # except StopIteration:
            #     raise RuntimeError(f"too many policies?")
            axs[1, col].plot(sfs[:, n, j], label=f'$\\pi_{n}, \\psi_{j}$')
          axs[1, col].legend()

      plt.tight_layout()
      self.wandb_log({f"{self.prefix}/sf-predictions-{wandb_suffix}": wandb.Image(fig)})
      plt.close(fig)
