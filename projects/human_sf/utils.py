from typing import Optional, Callable, NamedTuple
from acme.wrappers import observation_action_reward
import haiku as hk

import jax
import jax.numpy as jnp
import rlax

Array = jax.Array


def add_batch(nest, batch_size: Optional[int]):
  """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
  broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
  return jax.tree_util.tree_map(broadcast, nest)

def process_inputs(
    vision_fn: hk.Module,
    task_encoder: hk.Module,
    inputs: observation_action_reward.OAR):
  # convert action to onehot
  num_primitive_actions = inputs.observation['actions'].shape[-1]
  # [A, A] array
  actions = jax.nn.one_hot(
      inputs.observation['actions'],
      num_classes=num_primitive_actions)

  # convert everything to floats
  inputs = jax.tree_map(lambda x: x.astype(jnp.float32), inputs)

  #----------------------------
  # process actions
  #----------------------------
  # 1 -> A+1: primitive actions
  # rest: object 
  # each are [N, D] where N differs for action and object embeddings
  def mlp(x):
    x = hk.Linear(128, w_init=hk.initializers.TruncatedNormal())(x)
    x = hk.Linear(128)(jax.nn.relu(x))
    return x
  action_mlp = hk.to_module(mlp)('action_mlp')
  object_mlp = hk.to_module(mlp)('object_mlp')

  actions = action_mlp(actions)
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
  task = task_encoder(inputs.observation['task'])

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

