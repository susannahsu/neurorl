
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
from projects.human_sf import usfa_dyna_offtask as usfa
from projects.human_sf import object_q_learning

Array = acme_types.NestedArray
Config = usfa.Config

LARGE_NEGATIVE = -1e7

mask_predictions = usfa.mask_predictions
get_actor_core = functools.partial(
  object_q_learning.get_actor_core,
  extract_q_values=lambda preds: preds.q_values)


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

    ###########################
    # Setup transition function: ResNet
    ###########################
    def transition_fn(action: int, state: jax.Array):
      action_onehot = jax.nn.one_hot(
          action, num_classes=num_actions)
      assert action_onehot.ndim in (1, 2), "should be [A] or [B, A]"

      def _transition_fn(action_onehot, state):
        """ResNet transition model that scales gradient.

        Same tricks that MuZero uses."""
        # action: [A]
        # state: [D]
        new_state = muzero_mlps.Transition(
            channels=config.state_dim,
            num_blocks=config.transition_blocks)(
            action_onehot, state)
        new_state = scale_gradient(new_state, config.scale_grad)


        # STATE FEATURES at next time-step
        state_feature_logits = muzero_mlps.PredictionMlp(
          config.feature_layers,
          state_features_dim,
          name='state_features')(new_state)
        if config.binary_feature_loss:
          state_features = distrax.Bernoulli(
            logits=state_feature_logits).sample(seed=hk.next_rng_key())
          state_features = state_features.astype(new_state.dtype)
        else:
          state_features = state_feature_logits

        #----------------------------
        # discounts
        #----------------------------
        discount_logits = muzero_mlps.PredictionMlp(
          config.feature_layers, 1, name='discounts')(new_state)
        discount_logits = jnp.squeeze(discount_logits)
        discount = distrax.Bernoulli(
          logits=discount_logits).sample(seed=hk.next_rng_key())
        discount = discount.astype(new_state.dtype)

        #----------------------------
        # OBJECT MASK at next time-step
        #----------------------------
        action_mask_fn = muzero_mlps.PredictionMlp(
          config.feature_layers,
          num_actions, name='pred_action')
        action_mask_logits = action_mask_fn(new_state)
        action_mask = distrax.Bernoulli(
          logits=action_mask_logits).sample(seed=hk.next_rng_key())

        outputs = usfa.ModelOuputs(
          state=new_state,
          state_features=state_features,
          state_feature_logits=state_feature_logits,
          discount_logits=discount_logits,
          discount=discount,
          action_mask_logits=action_mask_logits,
          action_mask=action_mask)

        return outputs, new_state

      if action_onehot.ndim == 2:
        _transition_fn = jax.vmap(_transition_fn)
      return _transition_fn(action_onehot, state)
    transition_fn = hk.to_module(transition_fn)('transition_fn')

    ###################################
    # SF Head
    ###################################
    if config.head == 'independent':
      SfNetCls = usfa.IndependentSfHead
    elif config.head == 'monolithic':
      SfNetCls = usfa.MonolithicSfHead
    else:
      raise NotImplementedError

    sf_net_kwargs = dict(
        layers=config.sf_layers,
        state_features_dim=state_features_dim,
        num_actions=num_actions,
        policy_layers=config.policy_layers,
        combine_policy=config.combine_policy,
        activation=config.sf_activation,
        mlp_type=config.sf_mlp_type,
        out_init_value=config.out_init_value,
    )
    if config.sep_task_heads:
      sf_net = usfa.IndTaskHead(SfNetCls, sf_net_kwargs)
    else:
      sf_net = SfNetCls(**sf_net_kwargs)

    sf_head = usfa.SfGpiHead(
      num_actions=num_actions,
      sf_net=sf_net)


    return ObjectOrientedUsfaArch(
      torso=networks.OarTorso(
        num_actions=num_actions,
        vision_torso=networks.BabyAIVisionTorso(
          init_kernel=config.tile_size,
          relu_layer_0=config.relu_layer_0,
          conv_dim=config.final_conv_dim,
          out_dim=config.conv_flat_dim),
        output_fn=networks.TorsoOutput,
      ),
      memory=hk.LSTM(config.state_dim),
      transition_fn=transition_fn,
      eval_task_support=config.eval_task_support,
      sf_head=sf_head,
      policy_rep=config.policy_rep,
      )

  return usfa.make_mbrl_usfa_network(
    env_spec, make_core_module)
