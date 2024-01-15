import abc
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Union

import collections
import dataclasses
import pickle 
from absl import logging
from pprint import pprint

import dm_env
from dm_env import specs
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import operator
import tree
import math
import rlax
import wandb

import matplotlib.pyplot as plt
from acme.utils.observers import EnvLoopObserver

from td_agents.basics import ActorObserver, ActorState

Number = Union[int, float, np.float32, jnp.float32]


def flatten_dict(d, parent_key='', sep='_'):
  items = []
  for k, v in d.items():
      new_key = parent_key + sep + k if parent_key else k
      if isinstance(v, collections.MutableMapping):
          items.extend(flatten_dict(v, new_key, sep=sep).items())
      else:
          items.append((new_key, v))
  return dict(items)


def load_config(filename):
  with open(filename, 'rb') as fp:
    config = pickle.load(fp)
    logging.info(f'Loaded: {filename}')
    return config


def save_config(filename, config):
  with open(filename, 'wb') as fp:
      def fits(x):
        y = isinstance(x, str)
        y = y or isinstance(x, float)
        y = y or isinstance(x, int)
        y = y or isinstance(x, bool)
        return y
      new = {k:v for k,v in config.items() if fits(v)}
      pickle.dump(new, fp)
      logging.info(f'Saved: {filename}')



def merge_configs(
    dataclass_configs: Union[
      List[dataclasses.dataclass], dataclasses.dataclass],
    dict_configs: Union[
      List[Dict], Dict]
      ):

  if not isinstance(dataclass_configs, list):
    dataclass_configs = [dataclass_configs]
  if not isinstance(dict_configs, list):
    dict_configs = [dict_configs]

  everything = {}
  for tc in dataclass_configs:
    everything.update(tc.__dict__)

  for dc in dict_configs:
    everything.update(dc)

  config = dataclass_configs[0]
  for k, v in everything.items():
    setattr(config, k, v)

  return config

def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)


class LevelAvgObserver(EnvLoopObserver):
  """
  Compute some metric over many episodes. Has some helper variables to make tracking easier.

  Args:
      reset (int): Number of episodes before resetting the metric computation.
      prefix (str): Prefix to be added to the logged metrics.
      get_task_name (function): A function to get the task name. If None, a default function is used.
  
  Attributes:
      results (dict): A dictionary to store computed metrics for different tasks.
      task_name (str): The name of the current task.
      reset (int): Number of episodes before resetting the metric computation.
      prefix (str): Prefix to be added to the logged metrics.
      idx (int): Counter to keep track of the number of episodes.
      logging (bool): Flag to enable or disable logging.
  """
  def __init__(self, reset=100, prefix: str = '0.task', get_task_name=None):
    super(LevelAvgObserver, self).__init__()
    self.results = collections.defaultdict(list)
    self.task_name = None
    self.reset = reset
    self.prefix = prefix
    self.idx = 0
    self.logging = True
    if get_task_name is None:
      def get_task_name(env): return "Episode"
      logging.info(
          "WARNING: if multi-task, suggest setting `get_task_name` in `LevelAvgReturnObserver`. This will log separate statistics for each task.")
    self._get_task_name = get_task_name

  def current_task(self, env):
    """
    Get the name of the current task.

    Args:
        env: The environment.

    Returns:
        str: The name of the current task.
    """
    return self._get_task_name(env)

class LevelAvgReturnObserver(LevelAvgObserver):
  """Metric: Average return over many episodes"""

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Observes the initial state __after__ reset."""
    self.idx += 1
    self.task_name = self.current_task(env)
    self._episode_return = tree.map_structure(
      _generate_zeros_from_spec,
      env.reward_spec())

  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Records one environment step."""
    self._episode_return = tree.map_structure(
      operator.iadd,
      self._episode_return,
      timestep.reward if timestep.reward else 0.0)

    if timestep.last():
      self.results[self.task_name].append(self._episode_return)
      self.results['0.overall'].append(self._episode_return)

  def get_metrics(self) -> Dict[str, Number]:
    """Returns metrics collected for the current episode."""
    result = {}

    if self.idx % self.reset == 0:
      for key, returns in self.results.items():
        if not returns: continue
        avg = np.array(returns).mean()
        result[f'{self.prefix}/{key}/avg_return'] = float(avg)
        self.results[key] = []

      result['log_data'] = True

    return result

class AvgStateTDObserver(ActorObserver):
  def __init__(self,
               period=100,
               prefix: str = 'AvgStateTDObserver',
               discount: float = .99):
    super(AvgStateTDObserver, self).__init__()
    self.period = period
    self.prefix = prefix
    self.discount = discount
    self.idx = -1

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

  def flush_metrics(self) -> Dict[str, Number]:
    """Returns metrics collected for the current episode."""
    if not self.idx % self.period == 0:
      return

    ##################################
    # compute mean hidden
    ##################################
    recurrent_states = [s.recurrent_state for s in self.actor_states[1:]]
    lstm_hidden = jnp.stack([s.hidden for s in recurrent_states])  # [T, D]

    lstm_mean = lstm_hidden.mean(axis=1)

    ##################################
    # compute td-error
    ##################################
    # first prediction is empty (None)
    predictions = [s.predictions for s in self.actor_states[1:]]
    q_values = jnp.stack([p.q_values for p in predictions])  # [T, A]

    npreds = len(q_values)
    actions = jnp.stack(self.actions)[:npreds]
    q_values = rlax.batched_index(q_values, actions)  # [T]

    # ignore 0th (reset) time-step w/ 0 reward and last (terminal) time-step
    rewards = jnp.stack([t.reward for t in self.timesteps])[1:-1]

    q_pred = q_values[:-1]
    targets = rewards + self.discount*q_values[1:]

    td_error = q_pred - targets

    ##################################
    # log
    ##################################
    def get_wandb_obj(x, label: str, title: Optional[str] = None):
      t = np.arange(x)
      data = [[t, x] for (t, x) in zip(t, x)]
      table = wandb.Table(data=data, columns=["time", label])
      return wandb.plot.line(table, "time", label, title=title)

    wandb.log({
      f'{self.prefix}/lstm_mean': get_wandb_obj(lstm_mean, label='lstm'),
      f'{self.prefix}/td_error': get_wandb_obj(td_error, label='td_error')
    })

def episode_mean(x, mask):
  if len(mask.shape) < len(x.shape):
    nx = len(x.shape)
    nd = len(mask.shape)
    extra = nx - nd
    dims = list(range(nd, nd+extra))
    z = jnp.multiply(x, jnp.expand_dims(mask, dims))
  else:
    z = jnp.multiply(x, mask)
  return (z.sum(0))/(mask.sum(0)+1e-5)


def make_episode_mask(data= None, include_final=False, **kwargs):
  """Look at where have valid task data. Everything until 1 before final valid data counts towards task. Data.discount always ends two before final data. 
  e.g. if valid data is [x1, x2, x3, 0, 0], data.discount is [1,0,0,0,0]. So can use that to obtain masks.

  NOTE: should probably generalize but have not found need yet.
  Args:
      data (TYPE): Description
      include_final (bool, optional): if True, include all data. if False, include until 1 time-step before final data

  Returns:
      TYPE: Description
  """
  if data.discount.ndim == 2:
    T, B = data.discount.shape
    # for data [x1, x2, x3, 0, 0]
    if include_final:
      # return [1,1,1,0,0]
      return jnp.concatenate((jnp.ones((2, B)), data.discount[:-2]), axis=0)
    else:
      # return [1,1,0,0,0]
      return jnp.concatenate((jnp.ones((1, B)), data.discount[:-1]), axis=0)
  elif data.discount.ndim == 1:
    if include_final:
      return jnp.concatenate((jnp.ones((2,)), data.discount[:-2]), axis=0)
    else:
      return jnp.concatenate((jnp.ones((1,)), data.discount[:-1]), axis=0)
  else:
    raise NotImplementedError


def expand_tile_dim(x, size, axis=-1):
  """E.g. shape=[1,128] --> [1,10,128] if dim=1, size=10
  """
  ndims = len(x.shape)
  _axis = axis
  if axis < 0: # go AFTER -axis dims, e.g. x=[1,128], axis=-2 --> [1,10,128]
    axis += 1
    _axis = axis % ndims # to account for negative

  x = jnp.expand_dims(x, _axis)
  tiling = [1]*_axis + [size] + [1]*(ndims-_axis)
  return jnp.tile(x, tiling)


def array_from_fig(fig):
  # Save the figure to a numpy array
  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  img = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  return img

class Discretizer:
  def __init__(self,
               max_value: Union[float, int],
               num_bins: Optional[int] = None,
               step_size: Optional[int] = None,
               min_value: Optional[int] = None,
               clip_probs: bool = False,
               tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR):
    self._max_value = max_value
    self._min_value = min_value if min_value is not None else -max_value
    self._clip_probs = clip_probs
    if step_size is None:
      assert num_bins is not None
    else:
      num_bins = math.ceil((self._max_value-self._min_value)/step_size)+1

    self._num_bins = int(num_bins)
    self._tx_pair = tx_pair

  @property
  def num_bins(self):
    return self._num_bins

  def logits_to_scalar(self, logits):
     return self.probs_to_scalar(jax.nn.softmax(logits))

  def probs_to_scalar(self, probs):
    scalar = rlax.transform_from_2hot(
      probs=probs,
      min_value=self._min_value,
      max_value=self._max_value,
      num_bins=self._num_bins)
    unscaled_scalar = self._tx_pair.apply_inv(scalar)
    return unscaled_scalar

  def scalar_to_probs(self, scalar):
      scaled_scalar = self._tx_pair.apply(scalar)
      probs = rlax.transform_to_2hot(
      scalar=scaled_scalar,
      min_value=self._min_value,
      max_value=self._max_value,
      num_bins=self._num_bins)
      probs = jnp.clip(probs, 0, 1)  # for numerical stability
      return probs


@partial(jit, static_argnums=(1,))
def rolling_window(a, size: int):
    """Create rolling windows of a specified size from an input array.

    This function takes an input array 'a' and a 'size' parameter and creates rolling windows of the specified size from the input array.

    Args:
        a (array-like): The input array.
        size (int): The size of the rolling window.

    Returns:
        A new array containing rolling windows of 'a' with the specified size.
    """
    starts = jnp.arange(len(a) - size + 1)
    return jax.vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (size,)))(starts)


def scale_gradient(g: jax.Array, scale: float) -> jax.Array:
    """Scale the gradient.

    Args:
        g (_type_): Parameters that contain gradients.
        scale (float): Scale.

    Returns:
        Array: Parameters with scaled gradients.
    """
    return g * scale + jax.lax.stop_gradient(g) * (1.0 - scale)