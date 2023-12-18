from typing import Any, Dict, List, Optional, Sequence, Union

import collections
import pickle 
from absl import logging
from pprint import pprint

import dm_env
from dm_env import specs
import jax.numpy as jnp
import numpy as np
import operator
import tree

from acme.utils.observers import EnvLoopObserver

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


def update_config(config, strict: bool = True, **kwargs):
  for k, v in kwargs.items():
    if not hasattr(config, k):
      message = f"Attempting to set unknown attribute '{k}'"
      if strict:
        raise RuntimeError(message)
      else:
        logging.warning(message)
    setattr(config, k, v)


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)


class LevelAvgReturnObserver(EnvLoopObserver):
  """Metric: Average return over many episodes"""
  def __init__(self, reset=100, prefix :str = '0.task', get_task_name=None):
    super(LevelAvgReturnObserver, self).__init__()
    self.returns = collections.defaultdict(list)
    self.level = None
    self.reset = reset
    self.prefix = prefix
    self.idx = 0
    if get_task_name is None:
      get_task_name = lambda env: "Episode"
      logging.info("WARNING: if multi-task, suggest setting `get_task_name` in `LevelAvgReturnObserver`. This will log separate statistics for each task.")
    self._get_task_name = get_task_name

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Observes the initial state __after__ reset."""
    self.idx += 1
    if self.level is not None:
      self.returns[self.level].append(self._episode_return)
      self.returns['0.overall'].append(self._episode_return)

    self._episode_return = tree.map_structure(
      _generate_zeros_from_spec,
      env.reward_spec())
    self.level = self._get_task_name(env)


  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Records one environment step."""
    self._episode_return = tree.map_structure(
      operator.iadd,
      self._episode_return,
      timestep.reward)

  def get_metrics(self) -> Dict[str, Number]:
    """Returns metrics collected for the current episode."""
    result = {}

    if self.idx % self.reset == 0:
      # add latest (otherwise deleted)
      self.returns[self.level].append(self._episode_return)

      for key, returns in self.returns.items():
        if not returns: continue
        avg = np.array(returns).mean()
        result[f'{self.prefix}/{key}/avg_return'] = float(avg)
        self.returns[key] = []

      result['log_data'] = True

    return result


def episode_mean(x, mask):
  if len(mask.shape) < len(x.shape):
    nx = len(x.shape)
    nd = len(mask.shape)
    extra = nx - nd
    dims = list(range(nd, nd+extra))
    batch_loss = jnp.multiply(x, jnp.expand_dims(mask, dims))
  else:
    batch_loss = jnp.multiply(x, mask)
  return (batch_loss.sum(0))/(mask.sum(0)+1e-5)


def make_episode_mask(data, include_final=False, **kwargs):
  """Look at where have valid task data. Everything until 1 before final valid data counts towards task. Data.discount always ends two before final data. 
  e.g. if valid data is [x1, x2, x3, 0, 0], data.discount is [1,0,0,0,0]. So can use that to obtain masks.

  Args:
      data (TYPE): Description
      include_final (bool, optional): if True, include all data. if False, include until 1 time-step before final data

  Returns:
      TYPE: Description
  """
  T, B = data.discount.shape
  # for data [x1, x2, x3, 0, 0]
  if include_final:
    # return [1,1,1,0,0]
    return jnp.concatenate((jnp.ones((2, B)), data.discount[:-2]), axis=0)
  else:
    # return [1,1,0,0,0]
    return jnp.concatenate((jnp.ones((1, B)), data.discount[:-1]), axis=0)


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
