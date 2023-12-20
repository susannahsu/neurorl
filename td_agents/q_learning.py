
from typing import Optional, Tuple, Iterator, Optional, NamedTuple, Callable

import functools

from acme import specs
from acme.agents.jax import r2d2
from acme.jax.networks import base
from acme.jax import networks as networks_lib
from acme.wrappers import observation_action_reward
from acme.jax.networks import duelling
from acme import types as acme_types

import dataclasses
import haiku as hk
import jax
import jax.numpy as jnp
import rlax

import lib.networks as networks

from lib import utils 
from td_agents import basics

Array = acme_types.NestedArray

@dataclasses.dataclass
class Config(basics.Config):
  q_dim: int = 512

@dataclasses.dataclass
class R2D2LossFn(basics.RecurrentLossFn):

  loss: str = 'n_step_q_learning'

  def error(self, data, online_preds, online_state, target_preds, target_state, **kwargs):
    """R2D2 learning
    """
    # Get value-selector actions from online Q-values for double Q-learning.
    selector_actions = jnp.argmax(self.extract_q(online_preds), axis=-1)  # [T+1, B]
    # Preprocess discounts & rewards.
    discounts = (data.discount * self.discount).astype(self.extract_q(online_preds).dtype)
    rewards = data.reward
    rewards = rewards.astype(self.extract_q(online_preds).dtype)

    # Get N-step transformed TD error and loss.
    if self.loss == "transformed_n_step_q_learning":
      tx_pair = rlax.SIGNED_HYPERBOLIC_PAIR
    elif self.loss == "n_step_q_learning":
      tx_pair = rlax.IDENTITY_PAIR
    else:
      raise NotImplementedError(self.loss)

    batch_td_error_fn = jax.vmap(
        functools.partial(
            rlax.transformed_n_step_q_learning,
            n=self.bootstrap_n,
            tx_pair=tx_pair),
        in_axes=1,
        out_axes=1)

    batch_td_error = batch_td_error_fn(
        self.extract_q(online_preds)[:-1],  # [T+1] --> [T]
        data.action[:-1],    # [T+1] --> [T]
        self.extract_q(target_preds)[1:],  # [T+1] --> [T]
        selector_actions[1:],  # [T+1] --> [T]
        rewards[:-1],        # [T+1] --> [T]
        discounts[:-1])      # [T+1] --> [T]

    # average over {T} --> # [B]
    if self.mask_loss:
      # [T, B]
      episode_mask = utils.make_episode_mask(data, include_final=False)
      batch_loss = utils.episode_mean(
          x=(0.5 * jnp.square(batch_td_error)),
          mask=episode_mask[:-1])
    else:
      batch_loss = 0.5 * jnp.square(batch_td_error).mean(axis=0)

    metrics = {
        'z.q_mean': self.extract_q(online_preds).mean(),
        'z.q_var': self.extract_q(online_preds).var(),
        # 'z.q_max': online_preds.q_values.max(),
        # 'z.q_min': online_preds.q_values.min(),
        }

    return batch_td_error, batch_loss, metrics  # [T-1, B], [B]

class R2D2Arch(hk.RNNCore):
  """A duelling recurrent network for use with Atari observations as seen in R2D2.

  See https://openreview.net/forum?id=r1lyTjAqYX for more information.
  """

  def __init__(self,
               torso: networks.OarTorso,
               memory: hk.RNNCore,
               head: hk.Module,
               name: str = 'r2d2_arch'):
    super().__init__(name=name)
    self._torso = torso
    self._memory = memory
    self._head = head

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    embeddings = self._torso(inputs)  # [B, D+A+1]
    core_outputs, new_state = self._memory(embeddings, state)
    q_values = self._head(core_outputs)
    return q_values, new_state

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> hk.LSTMState:
    return self._memory.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    embeddings = hk.BatchApply(self._torso)(inputs)  # [T, B, D+A+1]
    core_outputs, new_states = hk.static_unroll(
      self._memory, embeddings, state)
    q_values = hk.BatchApply(self._head)(core_outputs)  # [T, B, A]
    return q_values, new_states

def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config,
        task_encoder: Callable[[Array], Array] = lambda obs: None
        ) -> r2d2.R2D2Networks:
  """Builds default R2D2 networks for Atari games."""

  num_actions = env_spec.actions.num_values

  def make_core_module() -> R2D2Arch:
    vision_torso = networks.AtariVisionTorso(
      out_dim=config.state_dim)

    observation_fn = networks.OarTorso(
      num_actions=num_actions,
      vision_torso=vision_torso,
      task_encoder=task_encoder,
    )

    return R2D2Arch(
      torso=observation_fn,
      memory=hk.LSTM(config.state_dim),
      head=duelling.DuellingMLP(num_actions,
                                hidden_sizes=[config.q_dim]))

  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)

