import functools

from typing import Optional, Tuple, Optional, Callable, Union, Any

import dataclasses
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import library.networks as networks

from td_agents import basics
from td_agents.basics import MbrlNetworks, make_mbrl_network
from td_agents import muzero

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

    self.process_inputs = functools.partial(
      utils.process_inputs,
      vision_fn=vision_fn,
      task_encoder=task_encoder,
      )

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

    state_input = self._vision_fn(inputs)  # [D+A+1]
    hidden, new_state = self._state_fn(state_input, state)
    root_outputs = self._root_pred_fn(hidden)

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
    state_input = hk.BatchApply(self._vision_fn)(inputs)
    all_hidden, new_state = hk.static_unroll(
        self._state_fn, state_input, state)
    root_output = hk.BatchApply(self._root_pred_fn)(all_hidden)

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

def get_actor_core(
    networks: MbrlNetworks,
    config: Config,
    discretizer: Optional[utils.Discretizer] = None,
    mcts_policy: Optional[
      Union[mctx.muzero_policy, mctx.gumbel_muzero_policy]] = None,
    evaluation: bool = True,
):
  """Returns ActorCore for MuZero."""

  if config.action_source == 'policy':
    select_action = functools.partial(
      policy_select_action,
      networks=networks,
      evaluation=evaluation)
  elif config.action_source == 'value':
    select_action = functools.partial(
      value_select_action,
      networks=networks,
      evaluation=evaluation,
      discretizer=discretizer,
      discount=config.discount)
  elif config.action_source == 'mcts':
    select_action = functools.partial(
      mcts_select_action,
      networks=networks,
      evaluation=evaluation,
      mcts_policy=mcts_policy,
      discretizer=discretizer,
      discount=config.discount)
  else:
    raise NotImplementedError(config.action_source)

  def init(
      rng: networks_lib.PRNGKey,
  ) -> basics.ActorState[actor_core_lib.RecurrentState]:
    rng, epsilon_rng, state_rng = jax.random.split(rng, 3)
    initial_core_state = networks.init_recurrent_state(
        state_rng)
    if evaluation:
      epsilon = config.evaluation_epsilon
    else:
      epsilon = jax.random.choice(
        epsilon_rng,
        np.logspace(
          start=config.epsilon_min,
          stop=config.epsilon_max,
          num=config.num_epsilons,
          base=config.epsilon_base))
    return basics.ActorState(
        rng=rng,
        epsilon=epsilon,
        recurrent_state=initial_core_state,
        prev_recurrent_state=initial_core_state)

  def get_extras(
          state: basics.ActorState[actor_core_lib.RecurrentState]) -> actor_core_lib.Extras:
    return {'core_state': state.prev_recurrent_state}

  return actor_core_lib.ActorCore(init=init,
                                  select_action=select_action,
                                  get_extras=get_extras)


def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config,
        task_encoder: Callable[[jax.Array], jax.Array] = lambda obs: None,
        **kwargs) -> MbrlNetworks:
  """Builds default MuZero networks for BabyAI tasks."""

  num_actions = env_spec.actions.num_values
  state_dim = config.state_dim

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

    def root_predictor(state: State):
      assert state.ndim in (1, 2), "should be [D] or [B, D]"

      def _root_predictor(state: State):
        state = root_base_transformation(state)
        policy_logits = root_policy_fn(state)
        value_logits = root_value_fn(state)

        return RootOutput(
            state=state,
            value_logits=value_logits,
            policy_logits=policy_logits,
        )
      if state.ndim == 2:
        _root_predictor = jax.vmap(_root_predictor)
      return _root_predictor(state)

    def model_predictor(state: State):
      assert state.ndim in (1, 2), "should be [D] or [B, D]"

      def _model_predictor(state: State):
        reward_logits = model_reward_fn(state)

        state = model_base_transformation(state)
        policy_logits = model_policy_fn(state)
        value_logits = model_value_fn(state)

        return ModelOutput(
            new_state=state,
            value_logits=value_logits,
            policy_logits=policy_logits,
            reward_logits=reward_logits,
        )
      if state.ndim == 2:
        _model_predictor = jax.vmap(_model_predictor)
      return _model_predictor(state)

    return ObjectOrientedMuZero(
        observation_fn=observation_fn,
        state_fn=state_fn,
        transition_fn=transition_fn,
        root_pred_fn=root_predictor,
        model_pred_fn=model_predictor)

  return make_mbrl_network(environment_spec=env_spec,
                      make_core_module=make_core_module,
                      **kwargs)
