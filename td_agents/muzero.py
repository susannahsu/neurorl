
from typing import Optional, Tuple, Optional, Callable, Union, Any
from absl import logging

import functools
from functools import partial
import chex
import distrax
import optax
import mctx
import reverb
import numpy as np

from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils as jax_utils
from acme.agents.jax import actor_core as actor_core_lib

import dataclasses
import haiku as hk
import jax
import jax.numpy as jnp
from jax import jit

import rlax
import library.networks as neural_networks

from library import utils 
from td_agents import basics
from library import muzero_mlps

BatchSize = int
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
  """Configuration options for MuZero agent."""
  discount: float = 0.997**4

  # Learner options
  batch_size: Optional[int] = 64
  trace_length: Optional[int] = 20
  sequence_period: Optional[int] = None

  warmup_steps: int = 1_000
  learning_rate_decay: float = .1
  lr_transition_steps: int = 1_000_000
  weight_decay: float = 1e-4
  lr_end_value: Optional[float] = 1e-5
  staircase_decay: bool = True

  tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR

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
  v_target_source: str = 'reanalyze'
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

  # Architecture
  state_dim: int = 256

  transition_blocks: int = 6  # number of resnet blocks
  prediction_blocks: int = 2  # number of resnet blocks
  seperate_model_nets: bool = False
  scale_grad: float = 0.5

  # actor hps
  action_source: str = 'policy'  # 'policy', 'mcts'

  root_policy_coef: float = 1.0
  root_value_coef: float = 0.25
  model_policy_coef: float = 10.0
  model_value_coef: float = 2.5
  model_reward_coef: float = 1.0

@dataclasses.dataclass
class MuZeroNetworks:
  """Network that can unroll state-fn and apply model over an input sequence."""
  init: InitFn
  apply: StateFn
  unroll: StateFn
  init_recurrent_state: Callable[[
      PRNGKey, Optional[BatchSize]], State]
  apply_model: ModelFn
  unroll_model: ModelFn


@chex.dataclass(frozen=True)
class RootOutput:
  state: jax.Array
  value_logits: jax.Array
  policy_logits: jax.Array


@chex.dataclass(frozen=True)
class ModelOutput:
  new_state: jax.Array
  reward_logits: jax.Array
  value_logits: jax.Array
  policy_logits: jax.Array


def muzero_optimizer_constr(config, initial_params=None):
  """Creates the optimizer for muzero.
  
  This includes:
  - a schedule where the learning rate goes up and then falls with an exponential decay
  - weight decay on the parameters
  - adam optimizer
  - max grad norm
  """

  ##########################
  # learning rate schedule
  ##########################
  if config.warmup_steps > 0:
    learning_rate = optax.warmup_exponential_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        end_value=config.lr_end_value,
        transition_steps=config.lr_transition_steps,
        decay_rate=config.learning_rate_decay,
        staircase=config.staircase_decay,
    )
  else:
    learning_rate = config.learning_rate

  ##########################
  # weight decay on parameters
  ##########################
  if config.weight_decay > 0.0:
    def decay_fn(module_name, name,
                  value): return True if name == "w" else False

    weight_decay_mask = hk.data_structures.map(decay_fn, initial_params)
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
        mask=weight_decay_mask,
    )
  else:
    optimizer = optax.adam(
        learning_rate=learning_rate, eps=config.adam_eps)

  ##########################
  # max grad norm
  ##########################
  if config.max_grad_norm:
    optimizer = optax.chain(optax.clip_by_global_norm(
        config.max_grad_norm), optimizer)
  return optimizer


@partial(jit, static_argnums=(1,))
def rolling_window(a, size: int):
    """Create rolling windows of a specified size from an input array.

    This function takes an input array 'a' and a 'size' parameter and creates rolling windows of the specified size from the input array.

    Args:
        a (array-like): The input array.
        size (int): The size of the rolling window.

    Returns:
        A new array containing rolling windows of 'a' with the specified size.
    """
    starts = jnp.arange(len(a) - size + 1)
    return jax.vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (size,)))(starts)

def model_step(params: networks_lib.Params,
               rng_key: jax.Array,
               action: jax.Array,
               state: jax.Array,
               discount: jax.Array,
               networks: MuZeroNetworks,
               discretizer: utils.Discretizer):
  """One simulation step in MCTS."""
  rng_key, model_key = jax.random.split(rng_key)
  model_output, next_state = networks.apply_model(
      params, model_key, state, action,
  )
  reward = discretizer.logits_to_scalar(model_output.reward_logits)
  value = discretizer.logits_to_scalar(model_output.value_logits)

  recurrent_fn_output = mctx.RecurrentFnOutput(
      reward=reward,
      discount=discount,
      prior_logits=model_output.policy_logits,
      value=value,
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

  simulation_steps : float = .5  # how many time-steps of simulation to learn model with
  reanalyze_ratio : float = .5  # how often to learn from MCTS data vs. experience
  mask_model: bool = True  # mask model outputs when out of bounds of data
  value_target_source: str = 'return'

  root_policy_coef: float = 1.0
  root_value_coef: float = 0.25
  model_policy_coef: float = 10.0
  model_value_coef: float = 2.5
  model_reward_coef: float = 1.0

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
    policy_probs_target, value_probs_target, reward_probs_target, returns, mcts_values = self.compute_target(
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
      policy_probs_target=policy_probs_target,
      value_probs_target=value_probs_target,
      reward_probs_target=reward_probs_target,
      returns=returns,
      **kwargs,
  )

  def compute_target(self,
                     data: jax.Array,
                     is_terminal_mask: jax.Array,
                     networks: MuZeroNetworks,
                     target_preds: RootOutput,
                     target_params: networks_lib.Params,
                     rng_key: networks_lib.PRNGKey,
                     **kwargs,
                     ):
    # ---------------
    # reward
    # ---------------
    reward_target = data.reward
    reward_probs_target = self.discretizer.scalar_to_probs(reward_target)
    reward_probs_target = jax.lax.stop_gradient(reward_probs_target)
    # ---------------
    # policy
    # ---------------
    # for policy, need initial policy + value estimates for each time-step
    # these will be used by MCTS
    target_values = self.discretizer.logits_to_scalar(
        target_preds.value_logits)
    roots = mctx.RootFnOutput(prior_logits=target_preds.policy_logits,
                              value=target_values,
                              embedding=self.state_from_preds(target_preds))

    invalid_actions = self.get_invalid_actions(
        batch_size=target_values.shape[0])

    # 1 step of policy improvement
    policy_targets_from_mcts = self.root_policy_coef > 0 or self.model_policy_coef
    targets_from_mcts = (self.value_target_source in ('mcts', 'reanalyze') or policy_targets_from_mcts)
    if targets_from_mcts:
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
    if policy_targets_from_mcts:
      policy_target = mcts_outputs.action_weights

      if invalid_actions is not None:
        valid_actions = 1 - invalid_actions
        uniform_policy = valid_actions/valid_actions.sum(-1, keepdims=True)

      random_policy_mask = jnp.broadcast_to(
          is_terminal_mask[:, None], policy_target.shape
      )
      policy_probs_target = jax.lax.select(
          random_policy_mask, uniform_policy, policy_target
      )
      policy_probs_target = jax.lax.stop_gradient(policy_probs_target)
    else:
      policy_probs_target = uniform_policy

    # ---------------
    # Values
    # ---------------
    # if not using MCTS, populate with fake
    discounts = (data.discount[:-1] *
                 self.discount).astype(target_values.dtype)
    if targets_from_mcts:
      mcts_values = mcts_outputs.search_tree.summary().value
    else:
      mcts_values = jnp.zeros_like(discounts)

    if self.value_target_source == 'mcts':
      returns = mcts_values
    elif self.value_target_source == 'return':
      returns = rlax.n_step_bootstrapped_returns(
          data.reward[:-1], discounts, target_values[1:], self.bootstrap_n)
    elif self.value_target_source == 'reanalyze':
      def return_fn(v): return rlax.n_step_bootstrapped_returns(
          data.reward[:-1], discounts, v[1:], self.bootstrap_n)
      dim_return = data.reward.shape[0] - 1
      rng_key, sample_key = jax.random.split(rng_key)

      reanalyze = distrax.Bernoulli(
          probs=self.reanalyze_ratio).sample(seed=rng_key)
      returns = jax.lax.cond(
          reanalyze > 0,
          lambda: mcts_values[:dim_return],
          lambda: return_fn(target_values))

    # Value targets for the absorbing state and the states after are 0.
    dim_return = returns.shape[0]
    value_target = jax.lax.select(
        is_terminal_mask[:dim_return], jnp.zeros_like(returns), returns)
    value_probs_target = self.discretizer.scalar_to_probs(value_target)
    value_probs_target = jax.lax.stop_gradient(value_probs_target)

    return policy_probs_target, value_probs_target, reward_probs_target, returns, mcts_values

  def learn(self,
            data: reverb.ReplaySample,
            networks: basics.NetworkFn,
            params: networks_lib.Params,
            in_episode: jax.Array,
            is_terminal_mask: jax.Array,
            online_preds: RootOutput,
            rng_key: networks_lib.PRNGKey,
            policy_probs_target: jax.Array,
            value_probs_target: jax.Array,
            reward_probs_target: jax.Array,
            returns: jax.Array,
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
        policy_probs_target (jax.Array): Target policy probabilities.
        value_probs_target (jax.Array): Target value probabilities.
        reward_probs_target (jax.Array): Target reward probabilities.
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
    dim_return = value_probs_target.shape[0]
    root_value_ce = jax.vmap(rlax.categorical_cross_entropy)(
        value_probs_target, online_preds.value_logits[:dim_return])
    # []
    raw_root_value_loss = utils.episode_mean(root_value_ce, in_episode[:dim_return])
    root_value_loss = self.root_value_coef*raw_root_value_loss

    # [T]
    root_policy_ce = policy_loss_fn(
        policy_probs_target, online_preds.policy_logits)
    # []
    raw_root_policy_loss = utils.episode_mean(root_policy_ce, in_episode)
    root_policy_loss = self.root_policy_coef*raw_root_policy_loss

    # for TD-error
    value_root_prediction = self.discretizer.logits_to_scalar(
        online_preds.value_logits[:dim_return])
    td_error = value_root_prediction - returns

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

    policy_model_target = jnp.concatenate(
        (policy_probs_target, uniform_policy))

    dummy_zeros = self.discretizer.scalar_to_probs(
        jnp.zeros(self.simulation_steps-1))
    reward_model_target = jnp.concatenate((reward_probs_target, dummy_zeros))

    if dim_return < nsteps:
      nz = self.simulation_steps+1
    else:
      nz = self.simulation_steps
    dummy_zeros = self.discretizer.scalar_to_probs(jnp.zeros(nz))
    value_model_target = jnp.concatenate((value_probs_target, dummy_zeros))

    # for every timestep t=0,...T,  we have predictions for t+1, ..., t+k where k = simulation_steps
    # use rolling window to create T x k prediction targets
    vmap_roll = jax.vmap(functools.partial(
        rolling_window, size=self.simulation_steps), 1, 2)
    policy_model_target = vmap_roll(policy_model_target[1:])  # [T, k, actions]
    value_model_target = vmap_roll(value_model_target[1:])    # [T, k, bins]
    reward_model_target = vmap_roll(reward_model_target)      # [T, k, bins]

    # ------------
    # get masks for losses
    # ------------
    # NOTE: npreds = nsteps + self.simulation_steps, number of predictions that will be made overall
    # NOTE: dim_return is the number of value predictions
    # if dim_return is LESS than number of predictions, then have extra target, so mask it
    extra_v = self.simulation_steps + int(dim_return < npreds)
    policy_mask = jnp.concatenate(
        (in_episode[1:], jnp.zeros(self.simulation_steps)))
    if self.mask_model:
      value_mask = jnp.concatenate(
          (in_episode[1:dim_return], jnp.zeros(extra_v)))
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
            reward_target_, value_target_, policy_target_,
            reward_mask_, value_mask_, policy_mask_):
      reward_ce = jax.vmap(rlax.categorical_cross_entropy)(
          reward_target_, model_outputs_.reward_logits)
      reward_loss = utils.episode_mean(reward_ce, reward_mask_)

      value_ce = jax.vmap(rlax.categorical_cross_entropy)(
          value_target_, model_outputs_.value_logits)
      value_loss = utils.episode_mean(value_ce, value_mask_)

      policy_ce = jax.vmap(rlax.categorical_cross_entropy)(
          policy_target_, model_outputs_.policy_logits)
      policy_loss = utils.episode_mean(policy_ce, policy_mask_)

      return reward_ce, value_ce, policy_ce, reward_loss, value_loss, policy_loss

    _ = [
        reward_model_ce,
        value_model_ce,
        policy_model_ce,
        model_reward_loss,
        model_value_loss,
        model_policy_loss] = jax.vmap(compute_losses)(
        model_outputs,
        reward_model_target, value_model_target, policy_model_target,
        reward_model_mask, value_model_mask, policy_model_mask)

    # all are []
    raw_model_policy_loss = utils.episode_mean(
        model_policy_loss, policy_mask[:nsteps])
    model_policy_loss = self.model_policy_coef * \
        raw_model_policy_loss
    raw_model_value_loss = utils.episode_mean(model_value_loss, value_mask[:nsteps])
    model_value_loss = self.model_value_coef * \
        raw_model_value_loss
    raw_model_reward_loss = utils.episode_mean(
        model_reward_loss, reward_mask[:nsteps])
    reward_loss = self.model_reward_coef * \
        raw_model_reward_loss

    total_loss = (
        reward_loss +
        root_value_loss + model_value_loss +
        root_policy_loss + model_policy_loss)

    metrics = {
        "0.0.total_loss": total_loss,
        "0.0.td-error": td_error,
        '0.1.policy_root_loss': raw_root_policy_loss,
        '0.1.policy_model_loss': raw_model_policy_loss,
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


def make_network(
        environment_spec: specs.EnvironmentSpec,
        make_core_module: Callable[[], hk.RNNCore]) -> MuZeroNetworks:
  """Builds a MuZeroNetworks from a hk.Module factory."""

  dummy_observation = jax_utils.zeros_like(environment_spec.observations)
  dummy_action = jnp.array(0)

  def make_unrollable_network_functions():
    network = make_core_module()

    def init() -> Tuple[networks_lib.NetworkOutput, State]:
      out, _ = network(dummy_observation, network.initial_state(None))
      return network.apply_model(out.state, dummy_action)

    apply = network.__call__
    return init, (apply,
                  network.unroll,
                  network.initial_state,
                  network.apply_model,
                  network.unroll_model)

  # Transform and unpack pure functions
  f = hk.multi_transform(make_unrollable_network_functions)
  apply, unroll, initial_state_fn, apply_model, unroll_model = f.apply

  def init_recurrent_state(key: jax_types.PRNGKey,
                           batch_size: Optional[int] =  None) -> State:
    no_params = None
    return initial_state_fn(no_params, key, batch_size)

  return MuZeroNetworks(
      init=f.init,
      apply=apply,
      unroll=unroll,
      apply_model=apply_model,
      unroll_model=unroll_model,
      init_recurrent_state=init_recurrent_state)

def policy_select_action(
        params: networks_lib.Params,
        observation: networks_lib.Observation,
        state: basics.ActorState[actor_core_lib.RecurrentState],
        networks: MuZeroNetworks,
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

def value_select_action(
        params: networks_lib.Params,
        observation: networks_lib.Observation,
        state: basics.ActorState[actor_core_lib.RecurrentState],
        networks: MuZeroNetworks,
        discretizer: utils.Discretizer,
        discount: float = .99,
        epsilon: float = .99,
        evaluation: bool = True):
  rng, policy_rng = jax.random.split(state.rng)

  preds, recurrent_state = networks.apply(
      params, policy_rng, observation, state.recurrent_state)

  #-----------------
  # function for computing Q-values for 1 action
  #-----------------
  def compute_q(state, action):
    """Q(s,a) = r(s,a) + \gamma* v(s')"""
    model_preds, _ = networks.apply_model(
        params, policy_rng, state, action)

    reward = discretizer.logits_to_scalar(
      model_preds.reward_logits)
    value = discretizer.logits_to_scalar(
      model_preds.value_logits)
    q_values = reward + discount*value
    return q_values

  # jit to make faster
  # vmap to do parallel over actions
  compute_q = jax.jit(jax.vmap(compute_q, in_axes=(None, 0)))


  #-----------------
  # compute Q-values for all actions
  #-----------------
  # create "batch" of actions
  nactions = preds.policy_logits.shape[-1]
  all_actions = jnp.arange(nactions)  # [A]

  # [A, 1]
  q_values = compute_q(
    recurrent_state.hidden, all_actions)
  # [A]
  q_values = jnp.squeeze(q_values, axis=-1)


  #-----------------
  # select action
  #-----------------
  if evaluation:
    action = jnp.argmax(q_values, axis=-1)
  else:
    rng, policy_rng = jax.random.split(rng)
    action = rlax.epsilon_greedy(epsilon).sample(policy_rng, q_values)

  return action, basics.ActorState(
      rng=rng,
      recurrent_state=recurrent_state,
      prev_recurrent_state=state.recurrent_state)

def mcts_select_action(
    mcts_policy: Optional[
        Union[mctx.muzero_policy, mctx.gumbel_muzero_policy]],
    discretizer: utils.Discretizer,
    params: networks_lib.Params,
    observation: networks_lib.Observation,
    state: basics.ActorState[actor_core_lib.RecurrentState],
    networks: MuZeroNetworks,
    discount: float = .99,
    evaluation: bool = True):
  rng, policy_rng = jax.random.split(state.rng)

  preds, recurrent_state = networks.apply(
    params, policy_rng, observation, state.recurrent_state)

  value = discretizer.logits_to_scalar(preds.value_logits)
  roots = mctx.RootFnOutput(prior_logits=preds.policy_logits,
                            value=value,
                            embedding=preds.state)


  # 1 step of policy improvement
  rng, improve_key = jax.random.split(rng)
  mcts_outputs = mcts_policy(
      params=params,
      rng_key=improve_key,
      root=roots,
      recurrent_fn=functools.partial(
          model_step,
          discount=jnp.full(value.shape, discount),
          networks=networks,
          discretizer=discretizer,
      ))

  policy_target = mcts_outputs.action_weights

  if evaluation:
    action = jnp.argmax(policy_target, axis=-1)
  else:
    action = jax.random.categorical(policy_rng, policy_target)

  return action, basics.ActorState(
      rng=rng,
      recurrent_state=recurrent_state,
      prev_recurrent_state=state.recurrent_state)

def get_actor_core(
    networks: MuZeroNetworks,
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
    state_input = hk.BatchApply(self._observation_fn)(inputs)
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

def scale_gradient(g: jax.Array, scale: float) -> jax.Array:
    """Scale the gradient.

    Args:
        g (_type_): Parameters that contain gradients.
        scale (float): Scale.

    Returns:
        Array: Parameters with scaled gradients.
    """
    return g * scale + jax.lax.stop_gradient(g) * (1.0 - scale)

def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config,
        task_encoder: Callable[[jax.Array], jax.Array] = lambda obs: None,
        **kwargs) -> MuZeroNetworks:
  """Builds default MuZero networks for BabyAI tasks."""

  num_actions = env_spec.actions.num_values
  state_dim = config.state_dim

  def make_core_module() -> MuZeroNetworks:

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
    state_fn = hk.LSTM(state_dim, name='state_lstm')

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
    root_vpi_base = muzero_mlps.ResMlp(
        config.prediction_blocks, name='pred_root_base')
    root_value_fn = muzero_mlps.PredictionMlp(
        (128, 32), config.num_bins, name='pred_root_value')
    root_policy_fn = muzero_mlps.PredictionMlp(
        (128, 32), num_actions, name='pred_root_policy')
    model_reward_fn = muzero_mlps.PredictionMlp(
        (32,), config.num_bins, name='pred_model_reward')

    if config.seperate_model_nets:
      # what is typically done
      model_vpi_base = muzero_mlps.ResMlp(
          config.prediction_blocks, name='root_model')
      model_value_fn = muzero_mlps.PredictionMlp(
          (128, 32), config.num_bins, name='pred_model_value')
      model_policy_fn = muzero_mlps.PredictionMlp(
          (128, 32), num_actions, name='pred_model_policy')
    else:
      model_value_fn = root_value_fn
      model_policy_fn = root_policy_fn
      model_vpi_base = root_vpi_base

    def root_predictor(state: State):
      assert state.ndim in (1, 2), "should be [D] or [B, D]"

      def _root_predictor(state: State):
        state = root_vpi_base(state)
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

        state = model_vpi_base(state)
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

    return MuZeroArch(
        observation_fn=observation_fn,
        state_fn=state_fn,
        transition_fn=transition_fn,
        root_pred_fn=root_predictor,
        model_pred_fn=model_predictor)

  return make_network(environment_spec=env_spec,
                      make_core_module=make_core_module,
                      **kwargs)
