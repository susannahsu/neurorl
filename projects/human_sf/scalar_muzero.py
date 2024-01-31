"""
Major changes from original MuZero:
- state is a tuple that includes task
  - this is useful for when you need to use the task for downstream processing. For example, if we assume reward is a dot-product between features and a task vector, you need access to that task vector in the model. One way we accomplish this is by having this be available in the "state" object.
- model now also predicts intermediary state-features
"""

from typing import Optional, Tuple, Optional, Callable, Union, Any, Sequence
from absl import logging

import functools
import chex
import distrax
import optax
import mctx
import reverb
import numpy as np

from acme import specs
from acme.jax import networks as networks_lib
from acme.agents.jax import actor_core as actor_core_lib

import dataclasses
import haiku as hk
import jax
import jax.numpy as jnp

import rlax
import library.networks as neural_networks

from library import utils
from library.utils import rolling_window
from library.utils import scale_gradient 
from td_agents import basics
from library import muzero_mlps
from td_agents.basics import MbrlNetworks, make_mbrl_network
from td_agents import muzero

from projects.human_sf import utils as project_utils

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


Config = muzero.Config
MuZeroArch = muzero.MuZeroArch
muzero_optimizer_constr = muzero.muzero_optimizer_constr


@dataclasses.dataclass
class Config(basics.Config):
  """Configuration options for MuZero agent."""

  # Architecture
  state_dim: int = 256
  reward_layers: Tuple[int] = (512, 512)
  policy_layers: Tuple[int] = (512, 512)
  value_layers: Tuple[int] = (512, 512)

  transition_blocks: int = 6  # number of resnet blocks
  prediction_blocks: int = 2  # number of resnet blocks
  seperate_model_nets: bool = False
  scale_grad: float = 0.5

  # actor hps
  action_source: str = 'policy'  # 'policy', 'mcts'

  # Learner options
  scalar_coef: float = 1e3
  root_policy_coef: float = 1.0
  root_value_coef: float = 0.25
  model_policy_coef: float = 10.0
  model_value_coef: float = 2.5
  model_reward_coef: float = 1.0
  model_features_coef: float = 1.0

  discount: float = 0.997**4
  tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR
  batch_size: Optional[int] = 64
  trace_length: Optional[int] = 20
  sequence_period: Optional[int] = None

  warmup_steps: int = 1_000
  learning_rate_decay: float = .1
  lr_transition_steps: int = 1_000_000
  weight_decay: float = 1e-4
  lr_end_value: Optional[float] = 1e-5
  staircase_decay: bool = True


  # Replay options
  samples_per_insert: float = 50.0
  min_replay_size: int = 1_000
  max_replay_size: int = 100_000

  # Loss hps
  num_bins: Optional[int] = 81  # number of bins for two-hot rep
  scalar_step_size: Optional[float] = None  # step size between bins
  # number of bins for two-hot rep  max_scalar_value: float = 10.0  # number of bins for two-hot rep
  max_scalar_value: float = 10.0
  # this interpolates between mcts output vs. observed return
  value_target_source: str = 'return'
  reanalyze_ratio: float = 0.5  # percent of time to use mcts vs. observed return
  mask_model: bool = True

  # MCTS general hps
  simulation_steps: int = 5
  num_simulations: int = 4
  max_sim_depth: Optional[int] = None
  q_normalize_epsilon: float = 0.01  # copied from `jax_muzero`
  # muzero_policy: str = 'gumbel_muzero'

  # MCTS muzero hps
  dirichlet_fraction: float = 0.25
  dirichlet_alpha: float = 0.3
  pb_c_init: float = 1.25
  pb_c_base: float = 19652
  temperature: float = 1.0

  # MCTS gumble_muzero hps
  maxvisit_init: int = 50
  gumbel_scale: float = 1.0


@chex.dataclass(frozen=True)
class RootOutput:
  state: jax.Array
  value: jax.Array
  policy_logits: jax.Array


@chex.dataclass(frozen=True)
class ModelOutput:
  new_state: jax.Array
  state_features: jax.Array
  reward: jax.Array
  value: jax.Array
  policy_logits: jax.Array

def model_step(
    params: networks_lib.Params,
    rng_key: jax.Array,
    action: jax.Array,
    state: jax.Array,
    discount: jax.Array,
    networks: MbrlNetworks,
    discretizer: utils.Discretizer):
  """One simulation step in MCTS."""
  del discretizer
  rng_key, model_key = jax.random.split(rng_key)
  model_output, next_state = networks.apply_model(
      params, model_key, state, action,
  )
  recurrent_fn_output = mctx.RecurrentFnOutput(
      reward=model_output.reward,
      discount=discount,
      prior_logits=model_output.policy_logits,
      value=model_output.value,
  )
  return recurrent_fn_output, next_state

@dataclasses.dataclass
class MuZeroLossFn(basics.RecurrentLossFn):
  """Computes MuZero loss. 
  
  Two main functions:
  - compute_targets: this computes the policy and value targets. For value it includes option to use reanalyze where some ratio of learning is from MCTS whereas the rest is from agent's experience.
  - afterwards loss is computed for root + model.

  Note: All the logic is done with data that only uses time-dimension. We use vmap to transform data before processing.

  Args:
    discretizer (utils.Discretizer): The discretizer used for data binning.
    mcts_policy (Union[mctx.muzero_policy, mctx.gumbel_muzero_policy]): The policy used in MCTS, which can be either 'muzero_policy' or 'gumbel_muzero_policy'.
    invalid_actions (Optional[jax.Array]): An optional array of invalid actions.
    simulation_steps (float): The number of time-steps for simulation during model learning.
    reanalyze_ratio (float): The ratio of learning from MCTS data vs. agent's experience.
    mask_model (bool): A flag indicating whether to mask model outputs when out of bounds of data.
    value_target_source (str): The source of the value target, which can be 'reanalyze'.
    model_policy_coef (float): The coefficient for the model's policy loss.
    model_value_coef (float): The coefficient for the model's value loss.
    model_reward_coef (float): The coefficient for the model's reward loss.
    root_value_coef (float): The coefficient for the root's value loss.
    root_policy_coef (float): The coefficient for the root's policy loss.
    state_from_preds (Callable[[RootOutput], jax.Array]): A callable function to extract the state from predictions.

  """
  discretizer: utils.Discretizer = utils.Discretizer(
    max_value=10, num_bins=101)
  mcts_policy: Union[mctx.muzero_policy,
                       mctx.gumbel_muzero_policy] = mctx.gumbel_muzero_policy
  invalid_actions: Optional[jax.Array] = None

  simulation_steps : float = 5  # how many time-steps of simulation to learn model with
  reanalyze_ratio : float = 0.5  # how often to learn from MCTS data vs. experience
  mask_model: bool = True  # mask model outputs when out of bounds of data
  value_target_source: str = 'return'

  scalar_coef: float = 1e3
  root_policy_coef: float = 1.0      # categorical
  model_policy_coef: float = 10.0    # categorical
  root_value_coef: float = 0.25    # scalar
  model_value_coef: float = 2.5    # scalar
  model_reward_coef: float = 1.0   # scalar
  model_features_coef: float = 1.0   # scalar

  state_from_preds: Callable[
    [RootOutput], jax.Array] = lambda preds: preds.state

  def error(self,
      data,
      online_preds,
      online_state,
      networks,
      params,
      target_preds,
      target_state,
      target_params,
      key_grad, **kwargs):

      learning_model = (
        self.model_policy_coef > 1e-8 or 
        self.model_value_coef > 1e-8 or
        self.model_reward_coef > 1e-8)

      T, B = data.discount.shape[:2]
      if learning_model:
        effective_bs = B*T*self.simulation_steps
        logging.warning(
            f"Given S={self.simulation_steps} simultions, the effective batch size for muzero model learning when B={B}, T={T}, and S={self.simulation_steps} is {effective_bs}.")
      else:
        effective_bs = B*T
        logging.warning(f"The effective batch size for muzero root learning when from MCTS when  B={B} and T={T} is {effective_bs}.")

      loss_fn = functools.partial(
        self.loss_fn,
        networks=networks,
        params=params,
        target_params=target_params,
        key_grad=key_grad)

      # vmap over batch dimension
      loss_fn = jax.vmap(loss_fn, 
        in_axes=(1, 1, None, 1, None), out_axes=0)

      # [B, T], [B], [B]
      td_error, total_loss, metrics = loss_fn(
        data,  # [T, B]
        online_preds,  #[T, B]
        online_state,  # [B]
        target_preds,  # [T, B]
        target_state,  # [B]
        )

      # parent call expects td_error to be [T, B]
      # [B, T] --> [T, B]
      td_error = jnp.transpose(td_error, axes=(1, 0))

      return td_error, total_loss, metrics

  def loss_fn(self,
            data,
            online_preds,
            online_state,
            target_preds,
            target_state,
            networks,
            params,
            target_params,
            key_grad, **kwargs):

    in_episode = utils.make_episode_mask(data, include_final=True)
    is_terminal_mask = utils.make_episode_mask(data, include_final=False) == 0.0

    # [T], [T/T-1], [T]
    key_grad, key = jax.random.split(key_grad)
    policy_target, value_target, reward_target, reward_return_target, mcts_values, value_target = self.compute_target(
        data=data,
        networks=networks,
        is_terminal_mask=is_terminal_mask,
        target_params=target_params,
        target_preds=target_preds,
        rng_key=key,
    )

    key_grad, key = jax.random.split(key_grad)
    return self.learn(
      data=data,
      networks=networks,
      params=params,
      in_episode=in_episode,
      is_terminal_mask=is_terminal_mask,
      online_preds=online_preds,
      rng_key=key,
      policy_target=policy_target,
      value_target=value_target,
      reward_target=reward_target,
      reward_return_target=reward_return_target,
      mcts_values=mcts_values,
      **kwargs,
  )

  def compute_target(self,
                     data: jax.Array,
                     is_terminal_mask: jax.Array,
                     networks: MbrlNetworks,
                     target_preds: RootOutput,
                     target_params: networks_lib.Params,
                     rng_key: networks_lib.PRNGKey,
                     **kwargs,
                     ):
    # ---------------
    # reward
    # ---------------
    reward_target = data.reward
    # ---------------
    # policy
    # ---------------
    # for policy, need initial policy + value estimates for each time-step
    # these will be used by MCTS
    target_values = target_preds.value
    roots = mctx.RootFnOutput(
      prior_logits=target_preds.policy_logits,
      value=target_values,
      embedding=self.state_from_preds(target_preds))

    invalid_actions = self.get_invalid_actions(
        batch_size=target_values.shape[0])

    # 1 step of policy improvement
    rng_key, improve_key = jax.random.split(rng_key)
    mcts_outputs = self.mcts_policy(
        params=target_params,
        rng_key=improve_key,
        root=roots,
        invalid_actions=invalid_actions,
        recurrent_fn=functools.partial(
            model_step,
            discount=jnp.full(target_values.shape, self.discount),
            networks=networks,
            discretizer=self.discretizer,
        ))

    num_actions = target_preds.policy_logits.shape[-1]
    uniform_policy = jnp.ones_like(target_preds.policy_logits) / num_actions

    #---------------------------
    # if learning policy using MCTS, add uniform actions to end
    # if not, just use uniform actions as targets
    #---------------------------
    policy_target = mcts_outputs.action_weights

    # ---------------
    # Values
    # ---------------
    discounts = (data.discount[:-1] *
                 self.discount).astype(target_values.dtype)
    mcts_values = mcts_outputs.search_tree.summary().value

    reward_return_target = rlax.n_step_bootstrapped_returns(
        data.reward[:-1], discounts, target_values[1:], self.bootstrap_n)
    value_target = reward_return_target

    # Value targets for the absorbing state and the states after are 0.
    num_value_preds = value_target.shape[0]
    value_target = jax.lax.select(
        is_terminal_mask[:num_value_preds], jnp.zeros_like(value_target), value_target)

    return policy_target, value_target, reward_target, reward_return_target, mcts_values, value_target

  def learn(self,
            data: reverb.ReplaySample,
            networks: basics.NetworkFn,
            params: networks_lib.Params,
            in_episode: jax.Array,
            is_terminal_mask: jax.Array,
            online_preds: RootOutput,
            rng_key: networks_lib.PRNGKey,
            policy_target: jax.Array,
            value_target: jax.Array,
            reward_target: jax.Array,
            reward_return_target: jax.Array,
            mcts_values: jax.Array,
            **kwargs,
            ):
    """
    Compute loss for MuZero.

    This function computes a loss using a combination of root losses
    and model losses. The steps are as follows:
    - compute losses for root predictions: value and policy
    - then create targets for the model. for starting time-step t, we will get predictions for t+1, t+2, t+3, .... We repeat this for starting at t+1, t+2, etc. We try to do this as much as we can in parallel. leads to T*K number of predictions.
    - we then compute losses for model predictions: reward, value, policy.

    Args:
        data (reverb.ReplaySample): Data sample from a replay buffer.
        networks (basics.NetworkFn): Network function for predictions.
        in_episode (jax.Array): Array indicating if each timestep is within an episode.
        is_terminal_mask (jax.Array): Array indicating if each timestep is a terminal state.
        online_preds (RootOutput): Model predictions for the current state.
        params (networks_lib.Params): Model parameters.
        rng_key (networks_lib.PRNGKey): Random number generator key.
        policy_target (jax.Array): Target policy probabilities.
        value_target (jax.Array): Target value probabilities.
        reward_target (jax.Array): Target reward probabilities.
        returns (jax.Array): value targets.

    Returns:
        tuple: A tuple containing:
            - td_error (jax.Array): Temporal difference error.
            - total_loss (float): Total loss.
            - metrics (dict): Dictionary containing various loss metrics.
    """
    nsteps = data.reward.shape[0]  # [T]

    policy_loss_fn = jax.vmap(rlax.categorical_cross_entropy)

    ###############################
    # Root losses
    ###############################
    # [T/T-1]
    num_value_preds = value_target.shape[0]
    value_root_prediction = online_preds.value[:num_value_preds]
    root_value_l2 = rlax.l2_loss(value_root_prediction, value_target)

    # []
    raw_root_value_loss = utils.episode_mean(root_value_l2, in_episode[:num_value_preds])
    root_value_coef = self.root_value_coef*self.scalar_coef
    root_value_loss = root_value_coef*raw_root_value_loss

    # [T]
    root_policy_ce = policy_loss_fn(
        policy_target, online_preds.policy_logits)
    # []
    raw_root_policy_loss = utils.episode_mean(root_policy_ce, in_episode)
    root_policy_loss = self.root_policy_coef*raw_root_policy_loss

    # for TD-error
    td_error = value_root_prediction - reward_return_target
    mcts_error = mcts_values[:num_value_preds] - reward_return_target

    # if not learning model
    if (self.model_policy_coef < 1e-8 and 
        self.model_value_coef < 1e-8 and
        self.model_reward_coef < 1e-8):

      total_loss = root_value_loss + root_policy_loss

      metrics = {
        "0.0.total_loss": total_loss,
        "0.0.td-error": td_error,
        '0.1.policy_root_loss': raw_root_policy_loss,
        '0.3.value_root_loss': raw_root_value_loss,
      }
      return td_error, total_loss, metrics

    ###############################
    # Model losses
    ###############################
    # ------------
    # prepare targets for model predictions
    # ------------
    # add dummy values to targets for out-of-bounds simulation predictions
    npreds = nsteps + self.simulation_steps
    num_actions = online_preds.policy_logits.shape[-1]
    uniform_policy = jnp.ones(
        (self.simulation_steps, num_actions)) / num_actions
    invalid_actions = self.get_invalid_actions(
        batch_size=self.simulation_steps)
    if invalid_actions is not None:
      valid_actions = 1 - invalid_actions
      uniform_policy = valid_actions/valid_actions.sum(-1, keepdims=True)

    # POLICY
    policy_model_target = jnp.concatenate(
        (policy_target[1:], uniform_policy))

    # STATE FEATURES
    state_features = data.observation.observation['state_features'][1:]
    dummy_zeros = jnp.zeros((self.simulation_steps, state_features.shape[-1]))
    state_features_target = jnp.concatenate((state_features, dummy_zeros))

    # REWARD
    dummy_zeros = jnp.zeros(self.simulation_steps-1)
    reward_model_target = jnp.concatenate((reward_target, dummy_zeros))

    # VALUE
    if num_value_preds < nsteps:
      nz = self.simulation_steps+1
    else:
      nz = self.simulation_steps
    dummy_zeros = jnp.zeros(nz)
    value_model_target = jnp.concatenate((value_target[1:], dummy_zeros))

    # for every timestep t=0,...T,  we have predictions for t+1, ..., t+k where k = simulation_steps
    # use rolling window to create T x k prediction targets
    roll = functools.partial(rolling_window, size=self.simulation_steps)
    vmap_roll = jax.vmap(roll, 1, 2)
    value_model_target = roll(value_model_target)    # [T, k]
    reward_model_target = roll(reward_model_target)      # [T, k]
    state_features_target = vmap_roll(state_features_target)  # [T, k, features]
    policy_model_target = vmap_roll(policy_model_target)  # [T, k, actions]

    # ------------
    # get masks for losses
    # ------------
    # NOTE: npreds = nsteps + self.simulation_steps, number of predictions that will be made overall
    # NOTE: num_value_preds is the number of value predictions
    # if num_value_preds is LESS than number of predictions, then have extra target, so mask it
    extra_v = self.simulation_steps + int(num_value_preds < npreds)
    policy_mask = jnp.concatenate(
        (in_episode[1:], jnp.zeros(self.simulation_steps)))
    if self.mask_model:
      value_mask = jnp.concatenate(
          (in_episode[1:num_value_preds], jnp.zeros(extra_v)))
      reward_mask = jnp.concatenate(
          (in_episode[:-1], jnp.zeros(self.simulation_steps)))
    else:
      reward_mask = value_mask = jnp.ones_like(policy_mask)
    reward_model_mask = rolling_window(reward_mask, self.simulation_steps)
    value_model_mask = rolling_window(value_mask, self.simulation_steps)
    policy_model_mask = rolling_window(policy_mask, self.simulation_steps)

    # ------------
    # get simulation actions
    # ------------
    rng_key, action_key = jax.random.split(rng_key)
    # unroll_actions = all_actions[start_i:start_i+simulation_steps]
    random_actions = jax.random.choice(
        action_key, num_actions, data.action.shape, replace=True)
    # for time-steps at the end of an episode, generate random actions from the last state
    simulation_actions = jax.lax.select(
        is_terminal_mask, random_actions, data.action)
    # expand simulations to account for model at end
    simulation_actions = jnp.concatenate(
        (simulation_actions,
         jnp.zeros(self.simulation_steps-1, dtype=simulation_actions.dtype))
    )
    simulation_actions = rolling_window(
        simulation_actions, self.simulation_steps)

    # ------------
    # unrolls the model from each time-step in parallel
    # ------------
    def model_unroll(key, state, actions):
      key, model_key = jax.random.split(key)
      model_output, _ = networks.unroll_model(
          params, model_key, state, actions)
      return model_output
    model_unroll = jax.vmap(model_unroll, in_axes=(0, 0, 0), out_axes=0)

    keys = jax.random.split(rng_key, num=len(policy_model_target)+1)
    model_keys = keys[1:]
    # T, |simulation_actions|, ...
    model_outputs = model_unroll(
        model_keys, self.state_from_preds(online_preds), simulation_actions,
    )

    # ------------
    # compute losses
    # ------------

    def compute_losses(
            model_outputs_,
            reward_target_, state_features_target_, value_target_, policy_target_,
            reward_mask_, value_mask_, policy_mask_):
      
      features_l2 = rlax.l2_loss(
          state_features_target_, model_outputs_.state_features)
      features_l2 = features_l2.mean(-1)
      features_loss = utils.episode_mean(features_l2, policy_mask_)

      reward_l2 = rlax.l2_loss(
          reward_target_, model_outputs_.reward)
      reward_loss = utils.episode_mean(reward_l2, reward_mask_)

      value_l2 = rlax.l2_loss(
          value_target_, model_outputs_.value)
      value_loss = utils.episode_mean(value_l2, value_mask_)

      policy_ce = jax.vmap(rlax.categorical_cross_entropy)(
          policy_target_, model_outputs_.policy_logits)
      policy_loss = utils.episode_mean(policy_ce, policy_mask_)

      return reward_loss, value_loss, policy_loss, features_loss

    _ = [
        model_reward_loss,
        model_value_loss,
        model_policy_loss,
        model_features_loss] = jax.vmap(compute_losses)(
        model_outputs,
        reward_model_target, state_features_target, value_model_target, policy_model_target,
        reward_model_mask, value_model_mask, policy_model_mask)

    # all are []
    raw_model_policy_loss = utils.episode_mean(
        model_policy_loss, policy_mask[:nsteps])
    model_policy_loss = self.model_policy_coef * \
        raw_model_policy_loss
    raw_model_value_loss = utils.episode_mean(model_value_loss, value_mask[:nsteps])
    model_value_loss = self.model_value_coef * \
        self.scalar_coef* raw_model_value_loss
    raw_model_reward_loss = utils.episode_mean(
        model_reward_loss, reward_mask[:nsteps])
    reward_loss = self.model_reward_coef * \
        self.scalar_coef * raw_model_reward_loss

    raw_model_features_loss = utils.episode_mean(
        model_features_loss, policy_mask[:nsteps])
    features_loss = self.scalar_coef * self.model_features_coef * \
        raw_model_features_loss


    total_loss = (
        reward_loss + features_loss + 
        root_value_loss + model_value_loss +
        root_policy_loss + model_policy_loss)

    metrics = {
        "0.0.total_loss": total_loss,
        "0.0.td-error-value": jnp.abs(td_error),
        "0.0.td-error-mcts": jnp.abs(mcts_error),
        '0.1.policy_root_loss': raw_root_policy_loss,
        '0.1.policy_model_loss': raw_model_policy_loss,
        '0.2.features_model_loss': raw_model_features_loss,  # T
        '0.2.reward_model_loss': raw_model_reward_loss,  # T
        '0.3.value_root_loss': raw_root_value_loss,
        '0.3.value_model_loss': raw_model_value_loss,  # T
    }

    return td_error, total_loss, metrics


  def get_invalid_actions(self, batch_size):
    """This computes invalid actions that match the batch_size."""
    if self.invalid_actions is None:
      return None
    if self.invalid_actions.ndim < 2:
      self.invalid_actions = jax.numpy.tile(
          self.invalid_actions, (batch_size, 1))
    return self.invalid_actions


def policy_select_action(
        params: networks_lib.Params,
        observation: networks_lib.Observation,
        state: basics.ActorState[actor_core_lib.RecurrentState],
        networks: MbrlNetworks,
        evaluation: bool = True):
  rng, policy_rng = jax.random.split(state.rng)

  logits, recurrent_state = networks.apply(
    params, policy_rng, observation, state.recurrent_state)

  if evaluation:
    action = jnp.argmax(logits.policy_logits, axis=-1)
  else:
    action = jax.random.categorical(policy_rng, logits.policy_logits)

  return action, basics.ActorState(
      rng=rng,
      recurrent_state=recurrent_state,
      prev_recurrent_state=state.recurrent_state)

def mcts_select_action(
    params: networks_lib.Params,
    observation: networks_lib.Observation,
    state: basics.ActorState[actor_core_lib.RecurrentState],
    discretizer: utils.Discretizer,
    mcts_policy: Optional[
        Union[mctx.muzero_policy, mctx.gumbel_muzero_policy]],
    networks: MbrlNetworks,
    discount: float = .99,
    evaluation: bool = True):

  rng, policy_rng = jax.random.split(state.rng)

  preds, recurrent_state = networks.apply(
    params, policy_rng, observation, state.recurrent_state)

  # MCTX assumes the following shapes
  # policy_logits [B, A]
  # value [B]
  # embedding [B, D]
  # here, we have B = 1
  # i.e MCTX assumes that input has batch dimension. add fake one.
  value = preds.value[None]
  root = mctx.RootFnOutput(
    prior_logits=preds.policy_logits[None],
    value=value,
    embedding=jax.tree_map(lambda x: x[None], preds.state))

  # 1 step of policy improvement
  rng, improve_key = jax.random.split(rng)
  mcts_outputs = mcts_policy(
      params=params,
      rng_key=improve_key,
      root=root,
      recurrent_fn=functools.partial(
          model_step,
          discount=jnp.full(value.shape, discount),
          networks=networks,
          discretizer=discretizer,
      ))

  # batch "0"
  policy_target = mcts_outputs.action_weights[0]

  if evaluation:
    action = jnp.argmax(policy_target, axis=-1)
  else:
    action = jax.random.categorical(policy_rng, policy_target)

  return action, basics.ActorState(
      rng=rng,
      recurrent_state=recurrent_state,
      prev_recurrent_state=state.recurrent_state)

def muzero_policy_act_mcts_eval(
    networks,
    config,
    discretizer,
    mcts_policy,
    evaluation: bool = True,
):
  """Returns ActorCore for MuZero."""

  if evaluation:
    select_action = functools.partial(mcts_select_action,
                                      networks=networks,
                                      evaluation=evaluation,
                                      mcts_policy=mcts_policy,
                                      discretizer=discretizer,
                                      discount=config.discount)
  else:
    select_action = functools.partial(policy_select_action,
                                      networks=networks,
                                      evaluation=evaluation)

  def init(rng):
    rng, state_rng = jax.random.split(rng, 2)
    initial_core_state = networks.init_recurrent_state(
        state_rng)

    return basics.ActorState(
        rng=rng,
        recurrent_state=initial_core_state,
        prev_recurrent_state=initial_core_state)

  def get_extras(state):
    return {'core_state': state.prev_recurrent_state}

  return actor_core_lib.ActorCore(init=init,
                                  select_action=select_action,
                                  get_extras=get_extras)

class MuZeroArch(hk.RNNCore):
  """MuZero Network Architecture.
  """

  def __init__(self,
               observation_fn: hk.Module,
               state_fn: hk.RNNCore,
               transition_fn: hk.RNNCore,
               root_pred_fn: RootFn,
               model_pred_fn: ModelFn,
               name='muzero_network'):
    super().__init__(name=name)
    self._observation_fn = observation_fn
    self._state_fn = state_fn
    self._transition_fn = transition_fn
    self._root_pred_fn = root_pred_fn
    self._model_pred_fn = model_pred_fn

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

    state_input = self._observation_fn(inputs)  # [D+A+1]
    task = inputs.observation['task'].astype(state_input.dtype)
    hidden, new_state = self._state_fn(state_input, state, task)
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
    state_input = hk.BatchApply(self._observation_fn)(inputs)
    task = inputs.observation['task'].astype(state_input.dtype)

    # at first step of unroll, place task into state
    state = project_utils.add_task_to_lstm_state(state, task[0])
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

class DotTaskFn(hk.Module):
  """A Dot-product Q-network."""

  def __init__(
      self,
      task_dim: int,
      hidden_sizes: Sequence[int],
      out_dim: int = 128,
      w_init: Optional[hk.initializers.Initializer] = None,
      name: str='dot_task'
  ):
    super().__init__(name=name)
    # self.task_dim = task_dim
    self.mlp = muzero_mlps.PredictionMlp(
          hidden_sizes, out_dim, name=f'{name}_mlp')
    self.task_linear = hk.nets.MLP((out_dim, out_dim))

  def __call__(self, state: jax.Array, task: jax.Array) -> jax.Array:
    """Forward pass of the network.
    
    Args:
        inputs (jnp.ndarray): Z
        w (jnp.ndarray): W

    Returns:
        jnp.ndarray: 2-D tensor of action values of shape [batch_size, num_actions]
    """
    task = self.task_linear(task)
    # Compute value & advantage for duelling.
    outputs = self.mlp(state)  # [C]
    outputs = jnp.sum(outputs*task, axis=-1)

    return outputs

def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config,
        task_encoder: Callable[[jax.Array], jax.Array] = lambda obs: None,
        **kwargs) -> MbrlNetworks:
  """Builds default MuZero networks for BabyAI tasks."""

  num_actions = env_spec.actions.num_values
  state_dim = config.state_dim
  task_dim = env_spec.observations.observation['task'].shape[-1]

  def make_core_module() -> MbrlNetworks:

    ###########################
    # Setup observation and state functions
    ###########################
    vision_torso = neural_networks.BabyAIVisionTorso(
        conv_dim=config.out_conv_dim,
        out_dim=config.state_dim)
    observation_fn = neural_networks.OarTorso(
        num_actions=num_actions,
        vision_torso=vision_torso,
        task_encoder=task_encoder,
    )

    state_fn = project_utils.TaskAwareRecurrentFn(
      core=hk.LSTM(state_dim, name='state_lstm'),
      task_dim=task_dim)

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
        new_hidden = muzero_mlps.Transition(
            channels=config.state_dim,
            num_blocks=config.transition_blocks)(
            action_onehot, state.hidden)
        new_hidden = scale_gradient(new_hidden, config.scale_grad)

        state._replace(hidden=new_hidden)

        return state, state

      if action_onehot.ndim == 2:
        _transition_fn = jax.vmap(_transition_fn)
      return _transition_fn(action_onehot, state)

    transition_fn = project_utils.TaskAwareRecurrentFn(
      core=hk.to_module(transition_fn)('transition_fn'),
      task_dim=task_dim)

    ###########################
    # Setup prediction functions: policy, value, reward
    ###########################
    root_base_transformation = muzero_mlps.ResMlp(
        config.prediction_blocks, name='pred_root_base')

    def make_prediction_mlp(layers, name):
        return DotTaskFn(
          hidden_sizes=layers,
          task_dim=task_dim,
          name=name
        )

    root_policy_fn = muzero_mlps.PredictionMlp(
        config.policy_layers, num_actions, name='pred_root_policy')

    root_value_fn = make_prediction_mlp(
        config.value_layers, name='pred_root_value')

    if config.seperate_model_nets:
      # what is typically done
      model_base_transformation = muzero_mlps.ResMlp(
          config.prediction_blocks, name='root_model')
      model_value_fn = make_prediction_mlp(
          config.value_layers, name='pred_model_value')
      model_policy_fn = muzero_mlps.PredictionMlp(
          config.policy_layers, num_actions, name='pred_model_policy')
    else:
      model_value_fn = root_value_fn
      model_policy_fn = root_policy_fn
      model_base_transformation = root_base_transformation

    def root_predictor(state: State):
      assert state.hidden.ndim in (1, 2), "should be [D] or [B, D]"

      def _root_predictor(state: State):
        state_rep = root_base_transformation(state.hidden)
        policy_logits = root_policy_fn(state_rep)
        value = root_value_fn(state_rep, state.task)

        return RootOutput(
            state=state,
            value=value,
            policy_logits=policy_logits,
        )
      if state.hidden.ndim == 2:
        _root_predictor = jax.vmap(_root_predictor)
      return _root_predictor(state)

    def model_predictor(state: State):
      assert state.hidden.ndim in (1, 2), "should be [D] or [B, D]"

      def _model_predictor(state: State):

        state_features = muzero_mlps.PredictionMlp(
          config.reward_layers,
          task_dim,
          name='state_features')(state.hidden)

        reward = (state_features*state.task).sum(-1)

        state_rep = model_base_transformation(state.hidden)
        policy_logits = model_policy_fn(state_rep)
        value = model_value_fn(state_rep, state.task)


        return ModelOutput(
            new_state=state,
            state_features=state_features,
            reward=reward,
            value=value,
            policy_logits=policy_logits,
        )
      if state.hidden.ndim == 2:
        _model_predictor = jax.vmap(_model_predictor)
      return _model_predictor(state)

    return MuZeroArch(
        observation_fn=observation_fn,
        state_fn=state_fn,
        transition_fn=transition_fn,
        root_pred_fn=hk.to_module(
          root_predictor)("root_predictor"),
        model_pred_fn=hk.to_module(
          model_predictor)("model_predictor"))

  return make_mbrl_network(environment_spec=env_spec,
                      make_core_module=make_core_module,
                      **kwargs)
