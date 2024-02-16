
import functools

from typing import Optional, Tuple, Optional, Callable


from acme import specs
from acme.agents.jax import r2d2
from acme.jax.networks import base
from acme.jax import networks as networks_lib
from acme.wrappers import observation_action_reward
from acme import types as acme_types
from acme.agents.jax import actor_core as actor_core_lib

import distrax
import dataclasses
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import library.networks as networks
from library.utils import episode_mean
from library.utils import make_episode_mask
from library.utils import rolling_window
from library.utils import scale_gradient
from library import muzero_mlps

from td_agents import basics
from projects.human_sf import cat_usfa as usfa
from projects.human_sf import object_q_learning

Array = acme_types.NestedArray
Config = usfa.Config

LARGE_NEGATIVE = -1e7

def mask_predictions(
    predictions: usfa.USFAPreds,
    action_mask: jax.Array):

  # mask out entries to large negative value
  mask = lambda x: jnp.where(action_mask, x, LARGE_NEGATIVE)
  q_values = mask(predictions.q_values)  # [A]

  # mask out entries to 0, vmap over cumulant dimension
  mask = lambda x: jnp.where(action_mask, x, 0.0)
  has_actions = len(predictions.sf) == 3 # [N, A, C]

  if has_actions:
    # vmap [N, C]
    mask = jax.vmap(mask)
    mask = jax.vmap(mask, 2, 2)
  else:
    # vmap [C]
    mask = jax.vmap(mask, 1, 1)
  sf = mask(predictions.sf)   # [..., A, Cumulants]

  return predictions._replace(
    q_values=q_values,
    sf=sf,
    action_mask=action_mask)


class ObjectOrientedUsfaArch(usfa.UsfaArch):
  """Universal Successor Feature Approximator."""

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState,  # [B, ...]
      evaluation: bool = False,
  ) -> Tuple[usfa.Predictions, hk.LSTMState]:

    predictions, new_state = super().__call__(inputs, state, evaluation)
    action_mask = inputs.observation['action_mask'].astype(predictions.sf.dtype)

    predictions = mask_predictions(predictions, action_mask)

    return predictions, new_state

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[usfa.Predictions, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""

    predictions, new_state = super().unroll(inputs, state)
    action_mask = inputs.observation['action_mask'].astype(predictions.sf.dtype)
    predictions = jax.vmap(jax.vmap(
      mask_predictions))(predictions, action_mask)

    return predictions, new_state

def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: usfa.Config) -> networks_lib.UnrollableNetwork:
  """Builds default USFA networks for Minigrid games."""

  num_primitive_actions = env_spec.actions.num_values
  num_possible_objects = env_spec.observations.observation['objects_mask'].shape[-1]
  num_actions = num_primitive_actions + num_possible_objects

  state_features_dim = env_spec.observations.observation['state_features'].shape[-1]

  def make_core_module() -> ObjectOrientedUsfaArch:

    if config.head == 'independent':
      SfNetCls = usfa.IndependentSfHead
    elif config.head == 'monolithic':
      SfNetCls = usfa.MonolithicSfHead
    else:
      raise NotImplementedError

    head = usfa.SfGpiHead(
      num_actions=num_actions,
      sf_net=SfNetCls(
        layers=config.sf_layers,
        state_features_dim=state_features_dim,
        num_actions=num_actions,
        policy_layers=config.policy_layers,
        combine_policy=config.combine_policy,
        activation=config.sf_activation,
        mlp_type=config.sf_mlp_type,
        out_init_value=config.out_init_value,
        ))

    return ObjectOrientedUsfaArch(
      torso=networks.OarTorso(
        num_actions=num_actions,
        vision_torso=networks.BabyAIVisionTorso(
          conv_dim=config.final_conv_dim,
          out_dim=config.conv_flat_dim),
        output_fn=networks.TorsoOutput,
      ),
      memory=hk.LSTM(config.state_dim),
      head=head)

  return usfa.make_mbrl_usfa_network(
    env_spec, make_core_module)
