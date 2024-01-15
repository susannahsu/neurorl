
from typing import Optional, Tuple, Optional, Callable

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

import library.networks as networks

from library import utils 
from td_agents import q_learning

Array = acme_types.NestedArray

@dataclasses.dataclass
class Config(q_learning.Config):
  q_dim: int = 512

# def make_object_action_inputs(
#     state: jax.Array,
#     objects: jax.Array,
#     ):
#   """
#   objects: [N, D], state: [D]. embed objects and concatenate to state.
#   """
#     action_embed = lambda x: hk.Linear(128, w_init=hk.initializers.TruncatedNormal())(x)
#   object_embed = lambda x: hk.Linear(128, w_init=hk.initializers.TruncatedNormal())(x)

#   # each are [B, N, D] where N differs for action and object embeddings
#   action_embeddings = hk.BatchApply(action_embed)(inputs.observation['actions'])
#   object_embeddings = hk.BatchApply(object_embed)(objects)
#   # option_inputs = jnp.concatenate((action_embeddings, object_embeddings), axis=-2)


class ObjectQHead(hk.Module):

  def __init__(self, primitive_head, object_head, name):
    super().__init__(name=name)
    self.primitive_head = primitive_head
    self.object_head = object_head

  def __call__(self, state: jax.Array, objects: jax.Array):
    import ipdb; ipdb.set_trace()
  # vmap concat over middle dimension to replicate concat across all "actions"
  # [B, D1] + [B, A, D2] --> [B, A, D1+D2]
  concat = lambda a, b: jnp.concatenate((a,b), axis=-1)
  concat = jax.vmap(in_axes=(None, 1), out_axes=1)(concat)

  q_ concat(state, objects)

class ObjectOrientedR2D2(hk.RNNCore):
  """A duelling recurrent network for use with Atari observations as seen in R2D2.

  See https://openreview.net/forum?id=r1lyTjAqYX for more information.
  """

  def __init__(self,
               vision_fn: networks.OarTorso,
               memory: hk.RNNCore,
               primitive_qhead: hk.Module,
               object_qhead: hk.Module,
               task_encoder: Optional[hk.Module] = None,
               name: str = 'r2d2_arch'):
    super().__init__(name=name)
    self._vision_fn = vision_fn
    self._memory = memory
    self._primitive_qhead = primitive_qhead
    self._object_qhead = object_qhead
    self._task_encoder = task_encoder

  def process_inputs(self, inputs: observation_action_reward.OAR):
    # convert action to onehot
    prev_action = inputs.action
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
    action_embed = lambda x: hk.Linear(128, w_init=hk.initializers.TruncatedNormal())(x)
    object_embed = lambda x: hk.Linear(128, w_init=hk.initializers.TruncatedNormal())(x)

    # each are [N, D] where N differs for action and object embeddings
    actions = hk.BatchApply(action_embed)(actions)
    objects = hk.BatchApply(object_embed)(inputs.observation['actions'])

    #----------------------------
    # compute selected action
    #----------------------------
    import ipdb; ipdb.set_trace()
    all_actions = jnp.concatenate((actions, objects), axis=-2)
    chosen_action = rlax.batched_index(all_actions, prev_action)
   
    #----------------------------
    # image + task + reward
    #----------------------------
    image = inputs.observation['image']/255.0
    image = self._vision_fn(image)  # [D+A+1]

    # task
    task = self._task_encoder(inputs.observation['task'])

    # reward = jnp.tanh(inputs.reward)
    reward = jnp.expand_dims(inputs.reward, axis=-1)

    state_input = jnp.concatenate(
      (image, task, chosen_action, reward), axis=-1)
    return state_input, objects

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:

    state_input, objects = self.process_inputs(inputs)
    core_outputs, new_state = self._memory(state_input, state)

    import ipdb; ipdb.set_trace()

    q_values = self._object_qhead(core_outputs, objects)
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

    state_input, objects = hk.BatchApply(self.process_inputs)(inputs)  # [T, B, D+A+1]

    core_outputs, new_states = hk.static_unroll(
      self._memory, state_input, state)

    q_values = hk.BatchApply(self._head)(core_outputs, objects)  # [T, B, A]
    return q_values, new_states

def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config,
        task_encoder: Callable[[Array], Array] = lambda obs: None
        ) -> r2d2.R2D2Networks:
  """Builds default R2D2 networks for Atari games."""

  num_actions = env_spec.actions.num_values

  def make_core_module() -> ObjectOrientedR2D2:

    vision_fn = networks.BabyAIVisionTorso(
          conv_dim=0, out_dim=config.state_dim)

    def object_qhead(state: jax.Array, objects: jax.Array):
      import ipdb; ipdb.set_trace()
      #--------------
      # primitive actions
      #--------------
      primitive_qhead=hk.nets.MLP([config.q_dim, num_actions])
      primitive_qs = primitive_qhead(state)

      #--------------
      # objects actions
      #--------------
      object_qhead = hk.nets.MLP([config.q_dim, 1])

      # vmap concat over middle dimension to replicate concat across all "actions"
      # [D1] + [A, D2] --> [A, D1+D2]
      concat = lambda a, b: jnp.concatenate((a,b), axis=-1)
      concat = jax.vmap(in_axes=(None, 1), out_axes=1)(concat)

      object_inputs = concat(state, objects)
      object_qs = object_qhead(object_inputs)
      object_qs = jnp.squeeze(object_qs, axis=-1)

      qs = jnp.concatenate((primitive_qs, object_qs))
      return qs

    object_qhead = hk.to_module(object_qhead)("object_qhead")


    return ObjectOrientedR2D2(
      torso=vision_fn,
      memory=hk.LSTM(config.state_dim),
      task_encoder=task_encoder,
      object_qhead=object_qhead,
      )


  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)

