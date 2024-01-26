"""
Only difference 
- Before: Q = f_q(s, w)
- Now: Q = dot (f_q(s, w), w)
"""

from typing import Optional, Tuple, Optional, Callable, Sequence

from acme import specs
from acme.agents.jax import r2d2
from acme.jax.networks import base
from acme.jax import networks as networks_lib
from acme.wrappers import observation_action_reward
from acme import types as acme_types

import haiku as hk
import jax
import jax.numpy as jnp

import library.networks as networks

from td_agents import q_learning as q_basic

Array = acme_types.NestedArray

Config = q_basic.Config
R2D2LossFn = q_basic.R2D2LossFn

def concat(inputs: networks.TorsoOutput):
  pieces = (inputs.image, inputs.action, inputs.reward, inputs.task)
  return jnp.concatenate(pieces, axis=-1)


class DotQMlp(hk.Module):
  """A Dot-product Q-network."""

  def __init__(
      self,
      num_actions: int,
      hidden_sizes: Sequence[int],
      out_dim: int = 128,
      w_init: Optional[hk.initializers.Initializer] = None,
  ):
    super().__init__(name='dot_qhead')
    self.num_actions = num_actions
    self.out_dim = out_dim

    self.q_mlp = hk.nets.MLP([*hidden_sizes, num_actions*out_dim], w_init=w_init)
    self.task_linear = hk.Linear(out_dim=out_dim, with_bias=False)


  def __call__(self, state: jax.Array, task: jax.Array) -> jax.Array:
    """Forward pass of the network.
    
    Args:
        inputs (jnp.ndarray): Z
        w (jnp.ndarray): W

    Returns:
        jnp.ndarray: 2-D tensor of action values of shape [batch_size, num_actions]
    """

    task = self.task_linear(task)
    outputs = self.q_mlp(state)  # [A*C]
    # [A, C]
    outputs = jnp.reshape(outputs, [self.num_actions, self.out_dim])

    q_vals = jax.vmap(jnp.multiply, in_axes=(0, None), out_axes=0)(outputs, task)

    # [A]
    q_vals = q_vals.sum(-1)

    return q_vals

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

    outputs = self._torso(inputs)  # [B, D+A+1]
    core_outputs, new_state = self._memory(concat(outputs), state)

    task = inputs.observation['task'].astype(jnp.float32)
    q_values = self._head(core_outputs, task)

    return q_values, new_state

  def initial_state(self, batch_size: Optional[int], **unused_kwargs) -> hk.LSTMState:
    return self._memory.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""

    outputs = hk.BatchApply(self._torso)(inputs)  # [T, B, D+A+1]

    core_outputs, new_states = hk.static_unroll(
      self._memory, concat(outputs), state)

    task = inputs.observation['task'].astype(jnp.float32)
    q_values = hk.BatchApply(jax.vmap(self._head))(core_outputs, task)  # [T, B, A]

    return q_values, new_states

def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config,
        task_encoder: Callable[[Array], Array] = lambda obs: None
        ) -> r2d2.R2D2Networks:
  """Builds default R2D2 networks for Atari games."""

  num_actions = env_spec.actions.num_values
  task_dim = env_spec.observations.observation['task'].shape[-1]

  def make_core_module() -> R2D2Arch:
    vision_torso = networks.BabyAIVisionTorso(
          conv_dim=0, out_dim=config.state_dim)
    observation_fn = networks.OarTorso(
      num_actions=num_actions,
      vision_torso=vision_torso,
      task_encoder=task_encoder,
      output_fn=networks.struct_output
    )

    return R2D2Arch(
      torso=observation_fn,
      memory=hk.LSTM(config.state_dim),
      head=DotQMlp(
        num_actions=num_actions,
        hidden_sizes=[config.q_dim],
      ))

  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)

