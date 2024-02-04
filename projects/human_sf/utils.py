from typing import Optional, Callable, NamedTuple
from acme.wrappers import observation_action_reward
import haiku as hk

import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import rlax
import wandb

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