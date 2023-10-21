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
