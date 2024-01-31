
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
from projects.human_sf import usfa
from projects.human_sf import object_q_learning

Array = acme_types.NestedArray

get_actor_core = functools.partial(
  object_q_learning.get_actor_core,
  extract_q_values=lambda preds: preds.q_values)

def mask_predictions(
    predictions: usfa.USFAPreds,
    objects_mask: jax.Array):
  q_values = predictions.q_values  # [A]
  sf = predictions.sf  # [A, D]

  import ipdb; ipdb.set_trace()

  return predictions._replace(
    q_values=q_values,
    sf=sf,
  )

class ObjectOrientedUsfaArch(hk.RNNCore):
  """Universal Successor Feature Approximator."""

  def __init__(self,
               vision_fn: networks.OarTorso,
               memory: hk.RNNCore,
               sf_head: usfa.SfGpiHead,
               name: str = 'usfa_arch'):
    super().__init__(name=name)
    self._memory = memory
    self._sf_head = sf_head

    process_inputs = functools.partial(
      utils.process_inputs,
      vision_fn=vision_fn,
      )
    self.process_inputs = hk.to_module(process_inputs)(
      "process_inputs")

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState,  # [B, ...]
      evaluation: bool = False,
  ) -> Tuple[usfa.USFAPreds, hk.LSTMState]:

    image, _, _, _, objects_mask = self.process_inputs(inputs)
    core_outputs, new_state = self._memory(image, state)

    import ipdb; ipdb.set_trace()
    if evaluation:
      predictions = self._sf_head.evaluate(
        task=inputs.observation['task'],
        usfa_input=core_outputs,
        train_tasks=inputs.observation['train_tasks']
        )
    else:
      predictions = self._sf_head(
        usfa_input=core_outputs,
        task=inputs.observation['task'],
      )

    predictions = mask_predictions(predictions, objects_mask)

    return predictions, new_state

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> hk.LSTMState:
    return self._memory.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[usfa.USFAPreds, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    image, _, _, _, objects_mask = hk.BatchApply(
      jax.vmap(self.process_inputs))(inputs)  # [T, B, D+A+1]

    core_outputs, new_states = hk.static_unroll(
      self._memory, image, state)

    predictions = jax.vmap(jax.vmap(self._head))(
        core_outputs,                # [T, B, D]
        inputs.observation['task'],  # [T, B]
      )

    predictions = jax.vmap(jax.vmap(
      mask_predictions))(predictions, objects_mask)

    return predictions, new_states

def make_object_oriented_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: usfa.Config) -> networks_lib.UnrollableNetwork:
  """Builds default USFA networks for Minigrid games."""

  num_actions = env_spec.actions.num_values
  state_features_dim = env_spec.observations.observation['state_features'].shape[0]

  def make_core_module() -> ObjectOrientedUsfaArch:

    sf_net = usfa.MonolithicSfHead(
      layers=config.sf_layers,
      state_features_dim=state_features_dim,
      num_actions=num_actions,
      policy_layers=config.policy_layers,
      # combine_policy=config.combine_policy,
    )
    usfa_head = usfa.SfGpiHead(
      num_actions=num_actions,
      nsamples=config.nsamples,
      variance=config.variance,
      sf_net=sf_net,
      eval_task_support=config.eval_task_support)

    return ObjectOrientedUsfaArch(
      vision_fn=networks.BabyAIVisionTorso(
          conv_dim=16, out_dim=config.state_dim),
      memory=hk.LSTM(config.state_dim),
      head=usfa_head)

  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)
