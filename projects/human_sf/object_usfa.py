
import functools

from typing import Optional, Tuple, Optional, Callable


from acme import specs
from acme.agents.jax import r2d2
from acme.jax.networks import base
from acme.jax import networks as networks_lib
from acme.wrappers import observation_action_reward
from acme import types as acme_types
from acme.agents.jax import actor_core as actor_core_lib

import dataclasses
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import library.networks as networks

from td_agents import basics
from projects.human_sf import utils
from projects.human_sf import usfa_offtask as usfa
from projects.human_sf import object_q_learning

Array = acme_types.NestedArray
Config = usfa.Config

LARGE_NEGATIVE = -1e7

get_actor_core = functools.partial(
  object_q_learning.get_actor_core,
  extract_q_values=lambda preds: preds.q_values)

def make_action_mask(objects_mask, num_actions):
  num_primitive_actions = num_actions - objects_mask.shape[-1]
  action_mask = jnp.ones(num_primitive_actions)
  return jnp.concatenate((action_mask, objects_mask), axis=-1)

def mask_predictions(
    predictions: usfa.USFAPreds,
    objects_mask: jax.Array):

  num_actions = predictions.q_values.shape[-1]
  action_mask = make_action_mask(objects_mask, num_actions)

  mask = lambda x: jnp.where(action_mask, x, LARGE_NEGATIVE)

  q_values = mask(predictions.q_values)  # [A]

  # # vmap over dims 0 and 2
  # mask = jax.vmap(mask)
  # mask = jax.vmap(mask, 2, 2)
  # sf = mask(predictions.sf)   # [N, A, Cumulants]

  return predictions._replace(
    q_values=q_values,
    # sf=sf,
    action_mask=action_mask)

class ObjectOrientedUsfaArch(hk.RNNCore):
  """Universal Successor Feature Approximator."""

  def __init__(self,
               torso: networks.OarTorso,
               memory: hk.RNNCore,
               head: usfa.SfGpiHead,
               learning_support: str,
               name: str = 'usfa_arch'):
    super().__init__(name=name)
    self._torso = torso
    self._memory = memory
    self._head = head
    self._learning_support = learning_support

    # process_inputs = functools.partial(
    #   utils.process_inputs,
    #   observation_fn=observation_fn,
    #   )
    # self.process_inputs = hk.to_module(process_inputs)(
    #   "process_inputs")

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> hk.LSTMState:
    return self._memory.initial_state(batch_size)

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState,  # [B, ...]
      evaluation: bool = False,
  ) -> Tuple[usfa.USFAPreds, hk.LSTMState]:
    
    torso_outputs = self._torso(inputs)  # [D+A+1]
    context = inputs.observation['context'].astype(torso_outputs.image.dtype)
    memory_input = jnp.concatenate(
      (torso_outputs.image, torso_outputs.action, context), axis=-1)
    core_outputs, new_state = self._memory(memory_input, state)

    objects_mask = inputs.observation['objects_mask'].astype(core_outputs.dtype)
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

    predictions = mask_predictions(predictions, objects_mask)

    return predictions, new_state

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[usfa.USFAPreds, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""

    torso_outputs = hk.BatchApply(self._torso)(inputs)  # [T, B, D+A+1]
    context = inputs.observation['context'].astype(torso_outputs.image.dtype)
    memory_input = jnp.concatenate(
      (torso_outputs.image, torso_outputs.action, context), axis=-1)

    core_outputs, new_states = hk.static_unroll(
      self._memory, memory_input, state)

    if self._learning_support == 'train_tasks':
      # [T, B, N, D]
      support = inputs.observation['train_tasks']
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

    objects_mask = inputs.observation['objects_mask'].astype(core_outputs.dtype)
    predictions = jax.vmap(jax.vmap(
      mask_predictions))(predictions, objects_mask)

    return predictions, new_states

def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: usfa.Config) -> networks_lib.UnrollableNetwork:
  """Builds default USFA networks for Minigrid games."""

  num_primitive_actions = env_spec.actions.num_values
  num_possible_objects = env_spec.observations.observation[
    'objects_mask'].shape[-1]
  num_actions = num_primitive_actions + num_possible_objects

  state_features_dim = env_spec.observations.observation['state_features'].shape[-1]

  def make_core_module() -> ObjectOrientedUsfaArch:

    if config.head == 'independent':
      SfNetCls = usfa.IndependentSfHead
    elif config.head == 'monolithic':
      SfNetCls = usfa.MonolithicSfHead
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

    usfa_head = usfa.SfGpiHead(
      num_actions=num_actions,
      nsamples=config.nsamples,
      variance=config.variance,
      sf_net=sf_net,
      eval_task_support=config.eval_task_support)

    return ObjectOrientedUsfaArch(
      torso=networks.OarTorso(
        num_actions=num_actions,
        vision_torso=networks.BabyAIVisionTorso(
          conv_dim=config.final_conv_dim,
          out_dim=config.conv_flat_dim),
        output_fn=networks.TorsoOutput,
      ),
      memory=hk.LSTM(config.state_dim),
      head=usfa_head,
      learning_support=config.learning_support)

  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)
