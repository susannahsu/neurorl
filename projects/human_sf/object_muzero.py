"""
Main changes:
- predicting objects mask
- masking policy with objects mask (visible at current time-step)
"""
import functools

from typing import Optional, Tuple, Optional, Callable, Union, Any

from acme import specs

import distrax
import haiku as hk
import jax
import jax.numpy as jnp

import library.networks as networks
from library.utils import scale_gradient 
from library import muzero_mlps

from td_agents import basics
from td_agents.basics import MbrlNetworks, make_mbrl_network

from projects.human_sf import muzero
from projects.human_sf import utils


BatchSize = int
PRNGKey = jax.random.KeyArray
Params = Any
Observation = jax.Array
Action = jax.Array
NetworkOutput = jax.Array
State = jax.Array
RewardLogits = jax.Array
PolicyLogits = jax.Array
ValueLogits = jax.Array

InitFn = Callable[[PRNGKey], Params]
StateFn = Callable[[Params, PRNGKey, Observation, State],
                   Tuple[NetworkOutput, State]]
RootFn = Callable[[State], Tuple[State, PolicyLogits, ValueLogits]]
ModelFn = Callable[[State], Tuple[State, RewardLogits, PolicyLogits, ValueLogits]]

RootOutput = muzero.RootOutput
ModelOutput = muzero.ModelOutput

Config = muzero.Config

class ObjectOrientedMuZero(hk.RNNCore):
  """MuZero Network Architecture.
  """

  def __init__(self,
               vision_fn: hk.Module,
               state_fn: hk.RNNCore,
               transition_fn: hk.RNNCore,
               root_pred_fn: RootFn,
               model_pred_fn: ModelFn,
               task_encoder: Optional[hk.Module] = None,
               name='muzero_network'):
    super().__init__(name=name)
    self._vision_fn = vision_fn
    self._state_fn = state_fn
    self._transition_fn = transition_fn
    self._root_pred_fn = root_pred_fn
    self._model_pred_fn = model_pred_fn

    process_inputs = functools.partial(
      utils.process_inputs,
      vision_fn=vision_fn,
      task_encoder=task_encoder,
      )
    self.process_inputs = hk.to_module(process_inputs)(
      "process_inputs")

  def initial_state(self,
                    batch_size: Optional[int] = None,
                    **unused_kwargs) -> State:
    return self._state_fn.initial_state(batch_size)

  def __call__(
      self,
      inputs: jax.Array,  # [...]
      state: State  # [...]
  ) -> Tuple[RootOutput, State]:
    """Apply state function over input.

    In theory, this function can be applied to a batch but this has not been tested.

    Args:
        inputs (jax.Array): typically observation. [D]
        state (State): state to apply function to. [D]

    Returns:
        Tuple[RootOutput, State]: single muzero output and single new state.
    """

    image, task, reward, objects, objects_mask = self.process_inputs(inputs)
    state_input = jnp.concatenate((image, task, reward), axis=-1)

    hidden, new_state = self._state_fn(state_input, state)
    root_outputs = self._root_pred_fn(hidden, objects_mask)

    return root_outputs, new_state

  def unroll(
      self,
      inputs: jax.Array,  # [T, B, ...]
      state: State  # [B, ...]
  ) -> Tuple[RootOutput, State]:
    """Unroll state function over inputs.

    Args:
        inputs (jax.Array): typically observations. [T, B, ...]
        state (State): state to begin unroll at. [T, ...]

    Returns:
        Tuple[RootOutput, State]: muzero outputs and single new state.
    """
    # [T, B, D+A+1]
    image, task, reward, objects, objects_mask = hk.BatchApply(
      jax.vmap(self.process_inputs))(inputs)  # [T, B, D+A+1]
    state_input = jnp.concatenate((image, task, reward), axis=-1)

    all_hidden, new_state = hk.static_unroll(
        self._state_fn, state_input, state)
    root_output = hk.BatchApply(self._root_pred_fn)(all_hidden, objects_mask)

    return root_output, new_state

  def apply_model(
      self,
      state: State,  # [B, D]
      action: jnp.ndarray,  # [B]
  ) -> Tuple[ModelOutput, State]:
    """This applies the model to each element in the state, action vectors.

    Args:
        state (State): states. [B, D]
        action (jnp.ndarray): actions to take on states. [B]

    Returns:
        Tuple[ModelOutput, State]: muzero outputs and new states for 
          each state state action pair.
    """
    # [B, D], [B, D]
    hidden, new_state = self._transition_fn(action, state)
    model_output = self._model_pred_fn(hidden)
    return model_output, new_state

  def unroll_model(
      self,
      state: State,  # [D]
      action_sequence: jnp.ndarray,  # [T]
  ) -> Tuple[ModelOutput, State]:
    """This unrolls the model starting from the state and applying the 
      action sequence.

    Args:
        state (State): starting state. [D]
        action_sequence (jnp.ndarray): actions to unroll. [T]

    Returns:
        Tuple[muzero_types.ModelOutput, State]: muzero outputs and single new state.
    """
    # [T, D], [D]
    all_hidden, new_state = hk.static_unroll(
        self._transition_fn, action_sequence, state)

    # [T, D]
    model_output = self._model_pred_fn(all_hidden)

    # [T, D], [D]
    return model_output, new_state

def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config,
        task_encoder: Callable[[jax.Array], jax.Array] = lambda obs: None,
        **kwargs) -> MbrlNetworks:
  """Builds default MuZero networks for BabyAI tasks."""

  num_primitive_actions = env_spec.actions.num_values
  num_possible_objects = env_spec.observations.observation[
    'objects_mask'].shape[-1]
  num_actions = num_primitive_actions + num_possible_objects

  def make_core_module() -> MbrlNetworks:

    ###########################
    # Setup transition function: ResNet
    ###########################
    def transition_fn(action: int, state: State):
      action_onehot = jax.nn.one_hot(
          action, num_classes=num_actions)
      assert action_onehot.ndim in (1, 2), "should be [A] or [B, A]"

      def _transition_fn(action_onehot, state):
        """ResNet transition model that scales gradient."""
        # action: [A]
        # state: [D]
        out = muzero_mlps.Transition(
            channels=config.state_dim,
            num_blocks=config.transition_blocks)(
            action_onehot, state)
        out = scale_gradient(out, config.scale_grad)
        return out, out

      if action_onehot.ndim == 2:
        _transition_fn = jax.vmap(_transition_fn)
      return _transition_fn(action_onehot, state)
    transition_fn = hk.to_module(transition_fn)('transition_fn')

    ###########################
    # Setup prediction functions: policy, value, reward
    ###########################
    root_base_transformation = muzero_mlps.ResMlp(
        config.prediction_blocks, name='pred_root_base')
    root_value_fn = muzero_mlps.PredictionMlp(
        config.value_layers, config.num_bins, name='pred_root_value')
    root_policy_fn = muzero_mlps.PredictionMlp(
        config.policy_layers, num_actions, name='pred_root_policy')

    model_objects_mask_fn = muzero_mlps.PredictionMlp(
        config.policy_layers,
        num_possible_objects, name='pred_model_objects')
    model_reward_fn = muzero_mlps.PredictionMlp(
        config.reward_layers, config.num_bins, name='pred_model_reward')

    if config.seperate_model_nets:
      # what is typically done
      model_base_transformation = muzero_mlps.ResMlp(
          config.prediction_blocks, name='root_model')
      model_value_fn = muzero_mlps.PredictionMlp(
          config.value_layers, config.num_bins, name='pred_model_value')
      model_policy_fn = muzero_mlps.PredictionMlp(
          config.policy_layers, num_actions, name='pred_model_policy')
    else:
      model_value_fn = root_value_fn
      model_policy_fn = root_policy_fn
      model_base_transformation = root_base_transformation

    def make_action_mask(objects_mask):
      action_mask = jnp.ones(num_primitive_actions)
      return jnp.concatenate((action_mask, objects_mask), axis=-1)

    def root_predictor(state: State, objects_mask: jax.Array):
      assert state.ndim in (1, 2), "should be [D] or [B, D]"

      def _root_predictor(state: State, objects_mask: jax.Array):
        state = root_base_transformation(state)
        value_logits = root_value_fn(state)
        policy_logits = root_policy_fn(state)

        action_mask = make_action_mask(objects_mask)
        policy_logits = jnp.where(action_mask, policy_logits, -1e8)

        return RootOutput(
            state=state,
            value_logits=value_logits,
            policy_logits=policy_logits,
            action_mask=action_mask,
        )
      if state.ndim == 2:
        _root_predictor = jax.vmap(_root_predictor)
      return _root_predictor(state, objects_mask)

    def model_predictor(state: State):
      assert state.ndim in (1, 2), "should be [D] or [B, D]"

      def _model_predictor(state: State):
        reward_logits = model_reward_fn(state)

        state = model_base_transformation(state)
        policy_logits = model_policy_fn(state)
        value_logits = model_value_fn(state)

        objects_mask_logits = model_objects_mask_fn(state)
        objects_mask = distrax.Bernoulli(
          logits=objects_mask_logits).sample(
            seed=hk.next_rng_key())

        action_mask = make_action_mask(objects_mask)
        policy_logits = jnp.where(action_mask, policy_logits, -1e8)

        return ModelOutput(
            new_state=state,
            value_logits=value_logits,
            policy_logits=policy_logits,
            reward_logits=reward_logits,
            objects_mask_logits=objects_mask_logits,
        )
      if state.ndim == 2:
        _model_predictor = jax.vmap(_model_predictor)
      return _model_predictor(state)

    return ObjectOrientedMuZero(
        vision_fn=networks.BabyAIVisionTorso(
          conv_dim=config.out_conv_dim, out_dim=config.state_dim),
        state_fn=hk.LSTM(config.state_dim),
        task_encoder=task_encoder,
        transition_fn=transition_fn,
        root_pred_fn=root_predictor,
        model_pred_fn=model_predictor)

  return make_mbrl_network(environment_spec=env_spec,
                      make_core_module=make_core_module,
                      **kwargs)
