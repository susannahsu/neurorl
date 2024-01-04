
from typing import Optional, Tuple, Optional, Callable, Any

import functools

import chex
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
import numpy as np

import library.networks as neural_networks
from library import muzero_mlps

from library import utils 
from td_agents import basics
from td_agents import q_learning

BatchSize = int
Array = acme_types.NestedArray
PRNGKey = networks_lib.PRNGKey
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
RootFn = Callable[[State], Tuple[PolicyLogits, ValueLogits]]
ModelFn = Callable[[State], Tuple[RewardLogits, PolicyLogits, ValueLogits]]

@dataclasses.dataclass
class Config(basics.Config):

  # arch configs
  state_transform_dims: Tuple[int] = (256,)
  q_dim: int = 512
  transition_blocks: int = 2  # number of resnet blocks in model
  scale_grad: float = 0.5  # how to scale gradient in transition model

  rl_coeff: float = 1.0  # online RL loss coeff
  dyna_coeff: float = 1.0  # dyna RL loss coeff
  reward_coeff: float = 1.0  # reward model loss

  # contrastive loss configs
  labels_from_target_params: bool = False  # use target params for negatives/targets
  model_coeff: float = 1e-3  # state model loss
  num_negatives: int = 10
  simulation_steps: int = 5  # steps to use for model learning
  temperature: float = .1  # contrastive loss temperature


@chex.dataclass(frozen=True)
class Predictions:
  state: jax.Array
  q_values: jax.Array
  rewards: jax.Array

def unit(a): return (a / (1e-5+jnp.linalg.norm(a, axis=-1, keepdims=True)))
def dot(a, b): return jnp.sum(a[:, None] * b[None], axis=-1)

def contrast_loss(
    y_hat: jax.Array,
    y: jax.Array,
    neg: jax.Array, temperature: float = 1.0):

  y_logits = dot(y_hat, y)  # [N, N]
  neg_logits = (y_hat[:, None]*neg).sum(-1)  # [N, E]

  #[N, N+E]
  logits = jnp.concatenate((y_logits, neg_logits), axis=-1)
  logits = logits/temperature
  
  # Compute the softmax probabilities
  log_probs = jax.nn.log_softmax(logits)

  num_classes = y_hat.shape[0]   # N
  nlogits = log_probs.shape[-1]  # N+E
  labels = jax.nn.one_hot(jnp.arange(num_classes), num_classes=nlogits)
  loss = rlax.categorical_cross_entropy(labels, logits)

  return loss, y_logits, neg_logits


@dataclasses.dataclass
class ContrastiveDynaLossFn(q_learning.R2D2LossFn):
  """
  This loss function consistents of 3 loss functions:
  1. An Q-learning loss that uses real data
  2. A dyna Q-learning loss that uses imaginary data
  3. A contrastive learning loss for learning (a) state-model and (b) a reward-model
  """
  labels_from_target_params: bool = False
  num_negatives: int = 10
  simulation_steps: int = 5
  temperature: float = .1
  extract_q: Callable[[Array], Array] = lambda preds: preds.q_values

  rl_coeff: float = 1.0
  dyna_coeff: float = 1.0
  reward_coeff: float = 1.0
  model_coeff: float = 1e-3

  def error(self,
            data,
            online_preds,
            online_state,
            target_preds,
            target_state,
            networks,
            params,
            target_params,
            key_grad, **kwargs):

    T, B = data.discount.shape[:2]

    metrics = {}
    total_batch_td_error = jnp.zeros((T, B))
    total_batch_loss = jnp.zeros(B)

    assert self.rl_coeff > 0 or self.dyna_coeff > 0
    #==========================
    # Online Q-learning loss
    # =========================
    if self.rl_coeff > 0:
      assert T > 1, 'need at least 2 state for this'
      r2d2_batch_td_error, r2d2_batch_loss, r2d2_metrics = super().error(
          data=data,
          online_preds=online_preds,
          online_state=online_state,
          target_preds=target_preds,
          target_state=target_state,
          **kwargs)

      # [T-1, B] --> [T, B]
      r2d2_batch_td_error = jnp.concatenate(
        (r2d2_batch_td_error, jnp.zeros((1, B))))

      total_batch_td_error += r2d2_batch_td_error*self.rl_coeff
      total_batch_loss += r2d2_batch_loss*self.rl_coeff
      metrics.update(r2d2_metrics)

    # ==========================
    # Dyna Q-learning loss
    # =========================
    if self.dyna_coeff > 0:
      dyna_loss = functools.partial(
        self.dyna_loss,
        networks=networks,
        params=params,
        target_params=target_params,
        rng_key=key_grad,
        )
      # vmap over batch dimension. allow for valid mask computation inside.
      dyna_loss = jax.vmap(dyna_loss, 1, 0)
      # [B, T], [B], [B]
      dyna_td_error, dyna_batch_loss, dyna_metrics = dyna_loss(
            data,
            online_preds,
            target_preds)
      # [B, T] --> [T, B]
      dyna_td_error = jnp.transpose(dyna_td_error, axes=(1, 0))

      # update
      total_batch_td_error += dyna_td_error*self.rl_coeff
      total_batch_loss += dyna_batch_loss*self.rl_coeff
      metrics.update(dyna_metrics)

    #==========================
    # Contrastive Model loss
    # =========================
    if self.reward_coeff > 0 or self.model_coeff > 0:
      #------------------
      # prepare negatives, N for each data-point
      #------------------
      n_time_preds = T-1  # excluding last data-point. no labels
      total_negatives = n_time_preds*B*self.num_negatives

      # dim = (T-1)*B*N
      negative_idx = np.random.randint(n_time_preds*B, size=total_negatives)

      if self.labels_from_target_params:
        negatives_source = target_preds.state
      else:
        negatives_source = online_preds.state

      # flatten
      # [T, B, D] --> [T*B, D]
      negatives_source = negatives_source.reshape(-1, *negatives_source.shape[2:])

      # select negatives
      # [(T-1)*B*N, D]
      negatives = negatives_source[negative_idx]

      # [(T-1)*B*N, D] --> [T-1, B, N, D]
      negatives = negatives.reshape(n_time_preds, B, self.num_negatives, *negatives.shape[1:])

      model_loss_fn = functools.partial(
        self.contrastive_model_loss,
        networks=networks,
        params=params,
        rng_key=key_grad,
      )
      # vmap over batch dimension
      model_loss_fn = jax.vmap(model_loss_fn, 
        in_axes=1, out_axes=0)

      model_batch_loss, model_metrics = model_loss_fn(
        online_preds,
        target_preds,
        data.action,
        data.reward,
        negatives)

      total_batch_loss += model_batch_loss
      metrics.update(model_metrics)

    if self.rl_coeff > 0.0:
      # remove last time-step since invalid for RL loss
      total_batch_td_error = total_batch_td_error[:-1]

    return total_batch_td_error, total_batch_loss, metrics


  def contrastive_model_loss(
          self,
          online_preds: jax.Array,
          target_preds: jax.Array,
          actions: jax.Array,
          rewards: jax.Array,
          negatives: jax.Array,
          networks: basics.MbrlNetworks,
          params: networks_lib.Params,
          rng_key: networks_lib.PRNGKey,
          ):
    # ==========================
    # prepare contrastive model learning loss pieces
    # =========================
    # consider states s_1, s_2, s_3, s_4
    # + actions a_1, a_2, a_3, a_4
    # if k=simulation_steps=2, we're only going to use the 1st 2 time-steps
    # to predict s1 --> {s2, s3}, s2--{s3, s4}
    # by doing this, we don't have to worry about masking
    # we DO have to worry about re-using {s3, s4} in OTHER batches as sources

    #---------------------------
    # get starting states
    #---------------------------
    # e.g. s_1, s_2
    # T' = T - sim_steps
    # [T', D]
    assert self.simulation_steps < len(online_preds.state)
    start_states = jax.tree_map(
      lambda x:x[:-self.simulation_steps], online_preds.state)

    #---------------------------
    # get model targets (positives)
    #---------------------------
    # this will apply a rolling window over
    vmap_roll = jax.vmap(functools.partial(utils.rolling_window, size=self.simulation_steps), 1,2)

    # flat s_2, s3, s_4
    if self.labels_from_target_params:
      next_states = jax.tree_map(
          lambda x: x[1:], target_preds.state)
    else:
      next_states = jax.tree_map(
          lambda x:x[1:], online_preds.state)

    # rolling window over next states
    # e.g. {s_2, s3}, {s3, s4}
    # [T, D] --> [T', sim_steps, D]
    state_model_targets = vmap_roll(next_states)

    # ----------------
    # rolling window over negatives
    # ---------------
    # [T, extras, dim] --> [T', sim_steps, extras, dim]
    # vmap over negatives dimension
    negatives = jax.vmap(vmap_roll, 1, 2)(negatives)

    # [T] --> [T', sim_steps]
    model_actions = utils.rolling_window(
        actions[:-1], self.simulation_steps)
    # ------------
    # unrolls the model from each time-step in parallel
    # ------------
    def model_unroll(key, state, actions):
      key, model_key = jax.random.split(key)
      model_output, _ = networks.unroll_model(
          params, model_key, state, actions)
      return model_output

    model_unroll = jax.vmap(model_unroll, in_axes=(0, 0, 0), out_axes=0)
    keys = jax.random.split(rng_key, num=len(state_model_targets)+1)

    # [T', sim_steps, ...]
    model_outputs = model_unroll(
        keys[1:], start_states, model_actions,
    )


    # ==========================
    # prepare reward model learning loss pieces
    # =========================
    # [T', sim_steps]
    reward_targets = utils.rolling_window(rewards[:-1], self.simulation_steps)

    # ==========================
    # define loss function
    # ==========================
    def compute_losses(
        reward_preds_,
        reward_targets_,
        state_model_preds_,
        state_model_targets_,
        negatives_):

      reward_loss = rlax.l2_loss(
          reward_preds_, reward_targets_)

      model_loss_fn = functools.partial(
        contrast_loss, temperature=self.temperature)

      model_loss, pos_logits, neg_logits = model_loss_fn(
        state_model_preds_, state_model_targets_, negatives_)

      return reward_loss, model_loss, pos_logits, neg_logits

    # both are [T', sim_steps, D]
    state_model_preds = unit(model_outputs.state)
    state_model_targets = unit(state_model_targets)
    # [T', sim_steps, N, D]
    negatives = unit(negatives)

    reward_loss, model_loss, pos_logits, neg_logits = jax.vmap(compute_losses)(
          model_outputs.rewards,
          reward_targets,
          state_model_preds,
          state_model_targets,
          negatives)

    # mean over (a) time (b) number of predictions
    model_loss = model_loss.mean()
    reward_loss = reward_loss.mean()
    metrics = {
      "1.0.model_loss": model_loss,
      "1.0.reward_loss": reward_loss,
      '1.1.pos_dot': pos_logits.mean(),
      '1.1.neg_dot': neg_logits.mean(),
    }

    reward_loss = reward_loss*self.reward_coeff
    model_loss = model_loss*self.model_coeff
    total_loss = reward_loss + model_loss

    return total_loss, metrics


  def dyna_loss(
          self,
          data,
          online_preds,
          target_preds,
          networks,
          params,
          target_params,
          rng_key,
    ):
    """

    Logic is as follows:
    - Have a set of starting states
    - for each starting state, select a random action
    - for each (s,a) pair, get (r,s', Q') via model
    - do Q-learning where using Q' from model as targets
    """

    #-------------------------
    # construct starting state, action pairs
    # actions sampled uniformly
    #-------------------------
    # starting states
    # [B, D]
    start_states = online_preds.state

    # random actions for each state
    # [B, D]
    num_actions = online_preds.q_values.shape[-1]
    uniform_policy = jnp.ones_like(online_preds.q_values) / num_actions
    rng_key, sample_key = jax.random.split(rng_key)
    model_actions = jax.random.categorical(sample_key, uniform_policy)

    # -------------------------
    # unroll model 1-time-step into future
    # -------------------------
    rng_key, model_key = jax.random.split(rng_key)
    model_outputs, _ = networks.apply_model(
        params, model_key, start_states, model_actions,
    )

    rng_key, model_key = jax.random.split(rng_key)
    target_model_outputs, _ = networks.apply_model(
        target_params, model_key, target_preds.state, model_actions,
    )

    # -------------------------
    # compute double Q-learning learning loss
    # -------------------------
    # Preprocess discounts.
    discounts = (data.discount *
                 self.discount).astype(start_states.dtype)

    batch_td_error_fn = jax.vmap(rlax.double_q_learning)

    batch_td_error = batch_td_error_fn(
        online_preds.q_values, # q_tm1
        model_actions,  # a_tm1
        jax.lax.stop_gradient(model_outputs.rewards),  # r_t
        discounts,  # discount_t
        target_model_outputs.q_values,  # q_t_value
        model_outputs.q_values,  # q_t_selector
        )

    # average over {T} --> # [B]
    if self.mask_loss:
      # [B]
      episode_mask = utils.make_episode_mask(data, include_final=True)
      batch_loss = utils.episode_mean(
          x=(0.5 * jnp.square(batch_td_error)),
          mask=episode_mask)
    else:
      batch_loss = 0.5 * jnp.square(batch_td_error).mean(axis=0)

    metrics = {
      '1.dyna_q_loss': batch_loss.mean(),
      '1.dyna_td_error': jnp.abs(batch_td_error).mean(),
      'z.dyna_q_mean': model_outputs.q_values.mean(),
      'z.dyna_q_var': model_outputs.q_values.var(),
    }

    batch_td_error = batch_td_error*self.dyna_coeff
    batch_loss = batch_loss*self.dyna_coeff

    return batch_td_error, batch_loss, metrics  # [T-1, B], [B]

class DynaArch(hk.RNNCore):
  """Dyna Architecture.
  """

  def __init__(self,
               observation_fn: hk.Module,
               state_fn: hk.RNNCore,
               transition_fn: hk.RNNCore,
               prediction_fn: RootFn,
               name='dyna_network'):
    super().__init__(name=name)
    self._observation_fn = observation_fn
    self._state_fn = state_fn
    self._transition_fn = transition_fn
    self._prediction_fn = prediction_fn

  def initial_state(self,
                    batch_size: Optional[int] = None,
                    **unused_kwargs) -> State:
    return self._state_fn.initial_state(batch_size)

  def __call__(
      self,
      inputs: jax.Array,  # [...]
      state: State  # [...]
  ) -> Tuple[Predictions, State]:
    """Apply state function over input.

    In theory, this function can be applied to a batch but this has not been tested.

    Args:
        inputs (jax.Array): typically observation. [D]
        state (State): state to apply function to. [D]

    Returns:
        Tuple[Predictions, State]: single predictions and single new state.
    """

    state_input = self._observation_fn(inputs)  # [D+A+1]
    state, new_state = self._state_fn(state_input, state)
    predictions = self._prediction_fn(state)

    return predictions, new_state

  def unroll(
      self,
      inputs: jax.Array,  # [T, B, ...]
      state: State  # [B, ...]
  ) -> Tuple[Predictions, State]:
    """Unroll state function over inputs.

    Args:
        inputs (jax.Array): typically observations. [T, B, ...]
        state (State): state to begin unroll at. [T, ...]

    Returns:
        Tuple[Predictions, State]: all predictions and single new state.
    """
    # [T, B, D+A+1]
    state_input = hk.BatchApply(self._observation_fn)(inputs)
    all_state, new_state = hk.static_unroll(
        self._state_fn, state_input, state)
    predictions = hk.BatchApply(self._prediction_fn)(all_state)

    return predictions, new_state

  def apply_model(
      self,
      state: State,  # [B, D]
      action: jnp.ndarray,  # [B]
  ) -> Tuple[Predictions, State]:
    """This applies the model to each element in the state, action vectors.

    Args:
        state (State): states. [B, D]
        action (jnp.ndarray): actions to take on states. [B]

    Returns:
        Tuple[Predictions, State]: prediction + new state.
    """
    # [B, D], [B, D]
    state_rep, new_state = self._transition_fn(action, state)
    predictions = self._prediction_fn(state_rep)
    return predictions, new_state

  def unroll_model(
      self,
      state: State,  # [D]
      action_sequence: jnp.ndarray,  # [T]
  ) -> Tuple[Predictions, State]:
    """This unrolls the model starting from the state and applying the 
      action sequence.

    Args:
        state (State): starting state. [D]
        action_sequence (jnp.ndarray): actions to unroll. [T]

    Returns:
        Tuple[Predictions, State]: all state predictions and single final state.
    """
    # [T, D], [D]
    all_state_rep, new_state = hk.static_unroll(
        self._transition_fn, action_sequence, state)

    # [T, D]
    predictions = self._prediction_fn(all_state_rep)

    # [T, D], [D]
    return predictions, new_state

def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config,
        task_encoder: Callable[[jax.Array], jax.Array] = lambda obs: None,
        **kwargs) -> basics.MbrlNetworks:
  """Builds default MuZero networks for BabyAI tasks."""

  num_actions = env_spec.actions.num_values
  state_dim = config.state_dim

  def make_core_module() -> basics.MbrlNetworks:

    ###########################
    # Setup observation and state functions
    ###########################
    vision_torso = neural_networks.BabyAIVisionTorso(
        conv_dim=0, out_dim=config.state_dim)

    observation_fn = neural_networks.OarTorso(
        num_actions=num_actions,
        vision_torso=vision_torso,
        task_encoder=task_encoder,
    )
    transformation = lambda x: x
    if config.state_transform_dims:
      # project hidden before outputting
      transformation = hk.nets.MLP(config.state_transform_dims)
    state_fn = neural_networks.LstmStateTransform(
      state_dim,
      transformation = transformation,
      name='state_lstm'
    )

    ###########################
    # Setup transition function: ResNet
    ###########################
    def transition_fn(action: int, state: State):
      action_onehot = jax.nn.one_hot(
          action, num_classes=num_actions)
      assert action_onehot.ndim in (1, 2), "should be [A] or [B, A]"

      def _transition_fn(action_onehot, state):
        """ResNet transition model that scales gradient.

        Same trick that MuZero uses."""
        # action: [A]
        # state: [D]
        out = muzero_mlps.Transition(
            channels=config.state_dim,
            num_blocks=config.transition_blocks)(
            action_onehot, state)
        out = utils.scale_gradient(out, config.scale_grad)
        return out, out

      if action_onehot.ndim == 2:
        _transition_fn = jax.vmap(_transition_fn)
      return _transition_fn(action_onehot, state)
    transition_fn = hk.to_module(transition_fn)('transition_fn')

    ###########################
    # Setup prediction functions for Q-values + rewards
    ###########################
    def prediction_fn(state):
      q_values = duelling.DuellingMLP(
        num_actions, hidden_sizes=[config.q_dim])(state)

      rewards = hk.nets.MLP([config.q_dim, 1])(state)
      rewards = jnp.squeeze(rewards, axis=-1)

      return Predictions(
        state=state,
        rewards=rewards,
        q_values=q_values)
    prediction_fn = hk.to_module(prediction_fn)('prediction_fn')

    return DynaArch(
        observation_fn=observation_fn,
        state_fn=state_fn,
        transition_fn=transition_fn,
        prediction_fn=prediction_fn,
    )

  return basics.make_mbrl_network(
    environment_spec=env_spec,
    make_core_module=make_core_module,
    **kwargs)
