
from typing import Callable, Dict, NamedTuple, Tuple, Optional

import dataclasses
import dm_env
import functools
import jax
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt
import rlax
import numpy as np
import wandb

from acme import specs
from acme.agents.jax.r2d2 import actor as r2d2_actor
from acme.jax import networks as networks_lib
from acme.agents.jax import actor_core as actor_core_lib
from acme.wrappers import observation_action_reward
from library.utils import episode_mean
from library.utils import make_episode_mask
from library.utils import expand_tile_dim

from td_agents import basics

import library.networks as networks
from td_agents.basics import ActorObserver, ActorState

@dataclasses.dataclass
class Config(basics.Config):
  eval_task_support: str = "train"  # options:
  nsamples: int = 0  # no samples outside of train vector
  variance: float = 0.1

  final_conv_dim: int = 16
  conv_flat_dim: Optional[int] = 0
  sf_layers : Tuple[int]=(256, 256)
  policy_layers : Tuple[int]=(256, 256)
  combine_policy: str = 'sum'

  head: str = 'monolithic'
  sf_coeff: float = 1.0
  q_coeff: float = 0.0
  sf_loss: str = 'qlearning'

def cumulants_from_env(data, online_preds, online_state, target_preds, target_state):
  # [T, B, C]
  cumulants = data.observation.observation['state_features']

  # ASSUMPTION: s' has cumulants for (s,a, r)
  cumulants = cumulants[1:].astype(jnp.float32)
  return cumulants

def cumulants_from_preds(
  data,
  online_preds,
  online_state,
  target_preds,
  target_state,
  stop_grad=True,
  use_target=False):

  if use_target:
    cumulants = target_preds.state_feature
  else:
    cumulants = online_preds.state_feature
  if stop_grad:
    return jax.lax.stop_gradient(cumulants) # [T, B, C]
  else:
    return cumulants # [T, B, C]

def sample_gauss(mean: jax.Array,
                 var: float,
                 key,
                 nsamples: int):
  # gaussian (mean=mean, var=.1I)
  # mean = [D]
  assert nsamples >= 0
  if nsamples >= 1:
    mean = jnp.expand_dims(mean, -2) # [1, D]
    samples = jnp.tile(mean, [nsamples, 1])  # [N, D]
    dims = samples.shape # [N, D]
    samples =  samples + jnp.sqrt(var) * jax.random.normal(key, dims)
    samples = samples.astype(mean.dtype)
  else:
    samples = jnp.expand_dims(mean, axis=1) # [N, D]
  return samples

def get_actor_core(
    networks: basics.NetworkFn,
    config: Config,
    evaluation: bool = False,
    extract_q_values = lambda preds: preds.q_values
  ):
  """Returns ActorCore for R2D2."""

  def select_action(params: networks_lib.Params,
                    observation: networks_lib.Observation,
                    state: ActorState[actor_core_lib.RecurrentState]):
    rng, policy_rng = jax.random.split(state.rng)

    preds, recurrent_state = networks.apply(params, policy_rng, observation, state.recurrent_state, evaluation)

    q_values = extract_q_values(preds)

    epsilon = state.epsilon
    action = rlax.epsilon_greedy(epsilon).sample(policy_rng, q_values)
    return action, ActorState(
        rng=rng,
        epsilon=state.epsilon,
        step=state.step + 1,
        predictions=preds,
        recurrent_state=recurrent_state,
        prev_recurrent_state=state.recurrent_state)

  def init(
      rng: networks_lib.PRNGKey
  ) -> ActorState[actor_core_lib.RecurrentState]:
    rng, epsilon_rng, state_rng = jax.random.split(rng, 3)
    if not evaluation:
      epsilon = jax.random.choice(epsilon_rng,
                                  np.logspace(
                                    start=config.epsilon_min,
                                    stop=config.epsilon_max,
                                    num=config.num_epsilons,
                                    base=config.epsilon_base))
    else:
      epsilon = config.evaluation_epsilon
    initial_core_state = networks.init_recurrent_state(state_rng, None)
    return ActorState(
        rng=rng,
        epsilon=epsilon,
        step=0,
        recurrent_state=initial_core_state,
        prev_recurrent_state=initial_core_state)

  def get_extras(
      state: ActorState[actor_core_lib.RecurrentState]
  ) -> r2d2_actor.R2D2Extras:
    return {'core_state': state.prev_recurrent_state}

  return actor_core_lib.ActorCore(init=init, select_action=select_action,
                                  get_extras=get_extras)

@dataclasses.dataclass
class UsfaLossFn(basics.RecurrentLossFn):

  extract_cumulants: Callable = cumulants_from_env
  extract_task: Callable = lambda data: data.observation.observation['task']
  extract_task_dim: Callable = lambda x: x[:, :, 0]

  sf_coeff: float = 1.0
  q_coeff: float = 0.0
  loss_fn : str = 'qlearning'
  lambda_: float  = .9

  def error(self, data, online_preds, online_state, target_preds, target_state, **kwargs):
    # ======================================================
    # Prepare Data
    # ======================================================
    # all are [T+1, B, N, A, C]
    # N = num policies, A = actions, C = cumulant dim
    online_sf = online_preds.sf
    online_z = online_preds.policy
    target_sf = target_preds.sf
    # pseudo rewards, [T/T+1, B, C]
    cumulants = self.extract_cumulants(
      data=data, online_preds=online_preds, online_state=online_state,
      target_preds=target_preds, target_state=target_state)
    cumulants = cumulants.astype(online_sf.dtype)

    # Get selector actions from online Q-values for double Q-learning.
    online_q =  (online_sf*online_z).sum(axis=-1) # [T+1, B, N, A]
    selector_actions = jnp.argmax(online_q, axis=-1) # [T+1, B, N]
    online_actions = data.action # [T, B]

    # Preprocess discounts & rewards.
    discounts = (data.discount * self.discount).astype(online_q.dtype) # [T, B]

    # ======================================================
    # Prepare loss (via vmaps)
    # ======================================================
    def td_error_fn(
      online_sf_,
      online_actions_,
      target_sf_,
      selector_actions_,
      cumulants_,
      discounts_,
    ):
      if self.loss_fn == 'qlearning':
        error_fn = functools.partial(
                rlax.transformed_n_step_q_learning,
                n=self.bootstrap_n)
        # vmap over cumulant dimension (C), return in dim=3
        error_fn = jax.vmap(
          error_fn,
          in_axes=(2, None, 2, None, 1, None), out_axes=1)
        return error_fn(
          online_sf_,  # [T, A, C]
          online_actions_,  # [T]         
          target_sf_,  # [T, A, C]
          selector_actions_,  # [T]      
          cumulants_,  # [T, C]      
          discounts_)  # [T]         

      elif self.loss_fn == 'qlambda':
        error_fn = jax.vmap(
          functools.partial(
              rlax.q_lambda,
              lambda_=self.lambda_),
          in_axes=(2, None, 1, None, 2), out_axes=1)

        return error_fn(
          online_sf_,       # [T, A, C] (vmap 2)
          online_actions_,  # [T]       (vmap None)
          cumulants_,       # [T, C]    (vmap 1)
          discounts_,       # [T]       (vmap None)
          target_sf_,        # [T, A, C] (vmap 2)
        )
      else:
        raise NotImplementedError

    # vmap over batch dimension (B), return B in dim=1
    td_error_fn = jax.vmap(td_error_fn, in_axes=1, out_axes=1)

    # vmap over policy dimension (N), return N in dim=2
    td_error_fn = jax.vmap(td_error_fn, in_axes=(2, None, 2, 2, None, None), out_axes=2)

    # output = [0=T, 1=B, 2=N, 3=C]
    sf_td_error = td_error_fn(
      online_sf[:-1],  # [T, B, N, A, C] (vmap 2,1)
      online_actions[:-1],  # [T, B]          (vmap None,1)
      target_sf[1:],  # [T, B, N, A, C] (vmap 2,1)
      selector_actions[1:],  # [T, B, N]       (vmap 2,1)
      cumulants,  # [T-1, B, C]       (vmap None,1)
      discounts[:-1])  # [T, B]          (vmap None,1)


    #---------------------
    # Q-learning TD-error
    # needs to happen now because will be averaging over dim N
    #  below this.
    #---------------------
    task_td = self.extract_task_dim(sf_td_error)
    task_w = self.extract_task(data)[:-1]
    # [T, B, C]*[T, B, C] = [T, B]
    value_td_error = (task_td*task_w).sum(-1)

    #---------------------
    # Compute average loss for SFs
    #---------------------
    # [T, B, N, C] --> [T, B]
    # mean over cumulants, mean over # of policies
    sf_td_error = sf_td_error.mean(axis=3).mean(axis=2)

    if self.mask_loss:
      # [T, B]
      episode_mask = make_episode_mask(data, include_final=False)
      sf_loss = episode_mean(
        x=(0.5 * jnp.square(sf_td_error)),
        mask=episode_mask[:-1])
    else:
      sf_loss = (0.5 * jnp.square(sf_td_error)).mean(axis=(0,2,3))

    batch_loss = sf_loss*self.sf_coeff
    #---------------------
    # Compute average loss for Q-values
    #---------------------

    q_loss = jnp.zeros_like(batch_loss)
    if self.q_coeff > 0.0:
      if self.mask_loss:
        # [T, B]
        q_loss = episode_mean(
          x=(0.5 * jnp.square(value_td_error)),
          mask=episode_mask[:-1])
      else:
        q_loss = (0.5 * jnp.square(value_td_error)).mean(axis=(0,2,3))

      batch_loss += q_loss*self.q_coeff

    cumulant_reward = (cumulants*task_w).sum(-1)
    reward_error = (cumulant_reward - data.reward[:-1])
    metrics = {      '0.total_loss': batch_loss,
      f'0.1.sf_loss_{self.loss_fn}': sf_loss,
      '0.1.q_loss': q_loss,
      '2.q_td_error': jnp.abs(value_td_error),
      '2.sf_td_error': jnp.abs(sf_td_error),
      '3.cumulants': cumulants,
      '3.sf_mean': online_sf,
      '3.q_mean': online_q,
      '4.reward_error': reward_error,
      # '3.sf_var': online_sf.var(),
      }

    return sf_td_error, batch_loss, metrics # [T, B], [B]

class Observer(ActorObserver):
  def __init__(self,
               period=100,
               plot_success_only: bool = False,
               prefix: str = 'SFsObserver'):
    super(Observer, self).__init__()
    self.period = period
    self.prefix = prefix
    self.successes = 0
    self.idx = -1
    self.logging = True
    self.plot_success_only = plot_success_only

  def wandb_log(self, d: dict):
    if self.logging:
      if wandb.run is not None:
        wandb.log(d)
      else:
        print("WANDB is not initialized")
        self.logging = False
        self.period = np.inf

  def observe_first(self, state: ActorState, timestep: dm_env.TimeStep) -> None:
    """Observes the initial state and initial time-step.

    Usually state will be all zeros and time-step will be output of reset."""
    self.idx += 1

    # epsiode just ended, flush metrics if you want
    if self.idx > 0:
      self.flush_metrics()

    # start collecting metrics again
    self.actor_states = [state]
    self.timesteps = [timestep]
    self.actions = []

  def observe_action(self, state: ActorState, action: jax.Array) -> None:
    """Observe state and action that are due to observation of time-step.

    Should be state after previous time-step along"""
    self.actor_states.append(state)
    self.actions.append(action)

  def observe_timestep(self, timestep: dm_env.TimeStep) -> None:
    """Observe next.

    Should be time-step after selecting action"""
    self.timesteps.append(timestep)

  def flush_metrics(self) -> Dict[str, float]:
    """Returns metrics collected for the current episode."""
    rewards = jnp.stack([t.reward for t in self.timesteps])[1:]
    total_reward = rewards.sum()
    if self.plot_success_only and not total_reward > 0:
      return

    self.successes += 1

    count = self.successes if self.plot_success_only else self.idx

    if not count % self.period == 0:
      return
    if not self.logging:
      return

    tasks = [t.observation.observation['task'] for t in self.timesteps]
    tasks = np.stack(tasks)
    task = tasks[0]

    # NOTES:
    # Actor states: T, starting at 0
    # timesteps: T, starting at 0
    # action: T-1, starting at 0
    ##################################
    # successor features
    ##################################
    # first prediction is empty (None)
    # [T, N, A, C]
    predictions = [s.predictions for s in self.actor_states[1:]]
    sfs = jnp.stack([p.sf for p in predictions])
    sfs = sfs[:, 0]  # [T-1, A, C]

    npreds = len(sfs)
    actions = jnp.stack(self.actions)[:npreds]

    index = functools.partial(jnp.take_along_axis, axis=-1)
    index = jax.vmap(index, in_axes=(2, None), out_axes=2)

    sfs = index(sfs, actions[:, None])  # [T-1, 1, C]
    sfs = jnp.squeeze(sfs, axis=-2)  # [T-1, C]

    q_values = (sfs*tasks[:-1]).sum(-1)

    # ignore 0th (reset) time-step w/ 0 reward and last (terminal) time-step
    state_features = jnp.stack([t.observation.observation['state_features'] for t in self.timesteps])[1:]

    discounts = jnp.stack([t.discount for t in self.timesteps[1:]])*.99
    n_step_return = functools.partial(rlax.n_step_bootstrapped_returns, n=5)
    n_step_return = jax.vmap(n_step_return, in_axes=(1, None, 1), out_axes=1)
    features_returns = n_step_return(state_features, discounts, sfs)

    ndims = sfs.shape[1]
    group_size = 5
    for i in range(0, ndims, group_size):
        # Create a figure and axis
        fig, ax = plt.subplots()

        # Determine the end of the current group
        end = min(i + group_size, ndims)

        # Plot each dimension in the group
        nplotted = 0
        default_cycler = iter(plt.rcParams['axes.prop_cycle'])
        for j in range(i, end):
          # Define a color for this pair of lines
          color = next(default_cycler)['color']

          # Plot state_features with a dashed line style
          if state_features[:, j].sum() > 0:
            nplotted += 1

            # STATE FEATURES
            ax.plot(state_features[:, j], label=f'$\\phi - {j}$', linestyle='--', color=color)

            # STATE FEATURE RETURNS
            ax.plot(features_returns[:, j], label=f'$\\phi - {j} - R$', linestyle=':', color=color)

            # SUCCESSOR FEATURES
            ax.plot(sfs[:, j], label=f'$\\psi - {j}$', color=color)

            ax.plot(rewards, label='rewards', linestyle='--', color='grey')
            ax.plot(q_values, label='q_values', color='grey')

        # Add labels and title
        ax.set_xlabel('Time')
        # ax.set_ylabel('Value')
        ax.set_title(f"Successor Feature Prediction (Dimensions {i} to {end-1})")

        # Log the plot to wandb
        if nplotted > 0:
          ax.legend()
          self.wandb_log({f"{self.prefix}/sf-group-prediction-{i//group_size}": wandb.Image(fig)})

        # Close the plot
        plt.close(fig)

    ##################################
    # q-values
    ##################################
    # if total_reward.sum():
    #   q_values = (sfs*tasks[:-1]).sum(-1)
    #   # Create a figure and axis
    #   fig, ax = plt.subplots()
    #   ax.plot(q_values, label='q_values')
    #   ax.plot(rewards, label='rewards')

    #   # Add labels and title if necessary
    #   ax.set_xlabel('time')
    #   ax.set_title(f"reward prediction: {task}\nR={total_reward}")
    #   ax.legend()

    #   # Log the plot to wandb
    #   self.wandb_log({f"{self.prefix}/reward_prediction": wandb.Image(fig)})

    #   # Close the plot
    #   plt.close(fig)

    ##################################
    # get images
    ##################################
    # [T, H, W, C]
    frames = np.stack([t.observation.observation['image'] for t in self.timesteps])
    self.wandb_log({
      f'{self.prefix}/episode-{task}': [wandb.Image(frame) for frame in frames]})

class USFAPreds(NamedTuple):
  q_values: jnp.ndarray  # q-value
  sf: jnp.ndarray # successor features
  policy: jnp.ndarray  # policy vector
  task: jnp.ndarray  # task vector (potentially embedded)

class MonolithicSfHead(hk.Module):
  def __init__(self,
               layers: int,
               num_actions: int,
               state_features_dim: int,
               combine_policy: str = 'concat',
               policy_layers : Tuple[int]=(32,),
               name: Optional[str] = None):
    super(MonolithicSfHead, self).__init__(name=name)

    if policy_layers:
      self.policy_net = hk.nets.MLP(policy_layers, activate_final=True)
    else:
      self.policy_net = lambda x: x

    self.sf_net = hk.nets.MLP(
        tuple(layers)+(num_actions * state_features_dim,))

    self.num_actions = num_actions
    self.state_features_dim = state_features_dim
    self.combine_policy = combine_policy

  def __call__(self,
               sf_input: jnp.ndarray,
               policy: jnp.ndarray,
               task: jnp.ndarray) -> jnp.ndarray:
    """Compute successor features and q-valuesu
    
    Args:
        sf_input (jnp.ndarray): D_1
        policy (jnp.ndarray): C_2
        task (jnp.ndarray): D
    
    Returns:
        jnp.ndarray: 2-D tensor of action values of shape [batch_size, num_actions]
    """
    policy = self.policy_net(policy)

    dim = sf_input.shape[-1]
    linear = lambda x: hk.Linear(dim, with_bias=False)(x)
    if self.combine_policy == 'concat':
      sf_input = jnp.concatenate((sf_input, policy))  # 2D
    elif self.combine_policy == 'product':
      sf_input = linear(jax.nn.relu(sf_input))*linear(policy)
    elif self.combine_policy == 'sum':
      sf_input = linear(jax.nn.relu(sf_input))+linear(policy)
    else:
      raise NotImplementedError

    # [A * C]
    sf = self.sf_net(sf_input)
    # [A, C]
    sf = jnp.reshape(sf, (self.num_actions, self.state_features_dim))

    def dot(a, b): return jnp.sum(a*b).sum()

    # dot-product: A
    q_values = jax.vmap(
        dot, in_axes=(0, None), out_axes=0)(sf, task)

    assert q_values.shape[0] == self.num_actions, 'wrong shape'
    return sf, q_values

class IndependentSfHead(hk.Module):
  """
  Independent SF heads help with optimization.

  Inspired by 
  1. Modular Successor Feature Approximators: https://arxiv.org/abs/2301.12305
  2. Categorical Successor Feature Approximators: https://arxiv.org/abs/2310.15940

  """
  def __init__(
      self,
      layers: int,
      num_actions: int,
      state_features_dim: int,
      policy_layers : Tuple[int]=(),
      name: Optional[str] = None):
    super(IndependentSfHead, self).__init__(name=name)

    self.sf_net_factory = lambda: hk.nets.MLP(
        tuple(layers)+(num_actions,))

    self.policy_layers = policy_layers
    if policy_layers:
      self.policy_net_factory = lambda: hk.nets.MLP(policy_layers, activate_final=True)
    else:
      identity = lambda x: x 
      self.policy_net_factory = lambda: identity

    self.num_actions = num_actions
    self.state_features_dim = state_features_dim

  def __call__(self,
               sf_input: jnp.ndarray,
               policy: jnp.ndarray,
               task: jnp.ndarray) -> jnp.ndarray:
    """Compute successor features and q-valuesu
    
    Args:
        sf_input (jnp.ndarray): D
        policy (jnp.ndarray): C
        task (jnp.ndarray): D
    
    Returns:
        jnp.ndarray: 2-D tensor of action values of shape [batch_size, num_actions]
    """
    ndims = self.state_features_dim
    assert policy.shape[0] == ndims

    # [C, 1]
    policy = jnp.expand_dims(policy, axis=1)
    if self.policy_layers:
      # [[P], ..., [P]]
      policy = [self.policy_net_factory()(policy[idx]) for idx in range(ndims)]
      # [C, P]
      policy = jnp.stack(policy, axis=0)

    # Below, we make C copies of sf_input, break policy into C vectors of dimension 1,
    # and then concatenate each break policy unit to each sf_input copy.
    concat = lambda a, b: jnp.concatenate((a, b))
    concat = jax.vmap(concat, in_axes=(None, 0), out_axes=0)
    sf_inputs = concat(sf_input, policy)  # [C, D+D]

    # now we get sf-estimates for each policy dimension
    # [[A], ..., [A]]
    sf = [self.sf_net_factory()(sf_inputs[idx]) for idx in range(ndims)]
    # [A, C]
    sf = jnp.stack(sf, axis=1)

    def dot(a, b): return jnp.sum(a*b).sum()

    # dot-product: A
    q_values = jax.vmap(
        dot, in_axes=(0, None), out_axes=0)(sf, task)

    assert q_values.shape[0] == self.num_actions, 'wrong shape'
    return sf, q_values

class SfGpiHead(hk.Module):
  """Universal Successor Feature Approximator GPI head"""
  def __init__(self,
    num_actions: int,
    sf_net : hk.Module,
    nsamples: int=10,
    variance: Optional[float]=0.5,
    eval_task_support: str = 'train', 
    ):
    """Summary
    
    Args:
        num_actions (int): Description
        hidden_size (int, optional): hidden size of SF MLP network
        variance (float, optional): variances of sampling
        nsamples (int, optional): number of policies
        eval_task_support (bool, optional): include eval task in support
    
    Raises:
        NotImplementedError: Description
    """
    super(SfGpiHead, self).__init__()
    self.num_actions = num_actions
    self.var = variance
    self.nsamples = nsamples
    self.eval_task_support = eval_task_support
    self.sf_net = sf_net

  def __call__(self,
    usfa_input: jnp.ndarray,  # memory output (e.g. LSTM)
    task: jnp.ndarray,  # task vector
    ) -> USFAPreds:
    policy = task # 1-1 mapping during training
    # -----------------------
    # policies + embeddings
    # -----------------------
    if self.nsamples > 0:
      # sample N times: [D_w] --> [N+1, D_w]
      policy_samples = sample_gauss(
        mean=policy,
        var=self.var,
        key=hk.next_rng_key(),
        nsamples=self.nsamples)
      # combine samples with the original policy vector
      policy_base = jnp.expand_dims(policy, axis=-2) # [1, D_w]
      policies = jnp.concatenate((policy_base, policy_samples), axis=-2)  # [N+1, D_w]
    else:
      policies = jnp.expand_dims(policy, axis=-2) # [1, D_w]

    return self.sfgpi(
      usfa_input=usfa_input,
      policies=policies,
      task=task)

  def evaluate(self,
    task: jnp.ndarray,  # task vector
    usfa_input: jnp.ndarray,  # memory output (e.g. LSTM)
    train_tasks: jnp.ndarray,  # all train tasks
    ) -> USFAPreds:

    if self.eval_task_support == 'train':
      # [N, D]
      policies = train_tasks

    elif self.eval_task_support == 'eval':
      # [1, D]
      policies = jnp.expand_dims(task, axis=-2)

    elif self.eval_task_support == 'train_eval':
      task_expand = jnp.expand_dims(task, axis=-2)
      # [N+1, D]
      policies = jnp.concatenate((train_tasks, task_expand), axis=-2)
    else:
      raise RuntimeError(self.eval_task_support)

    preds = self.sfgpi(
      usfa_input=usfa_input, policies=policies, task=task)

    return preds

  def sfgpi(self,
    usfa_input: jnp.ndarray,
    policies: jnp.ndarray,
    task: jnp.ndarray) -> USFAPreds:
    """Summary
    
    Args:
        usfa_input (jnp.ndarray): D, typically rnn_output
        policies (jnp.ndarray): N x D
        task (jnp.ndarray): D
    Returns:
        USFAPreds: Description
    """

    sfs, q_values = jax.vmap(
      self.sf_net, in_axes=(None, 0, None), out_axes=0)(
        usfa_input,
        policies,
        task)

    # GPI
    # -----------------------
    # [N, A] --> [A]
    q_values = jnp.max(q_values, axis=-2)

    policies = expand_tile_dim(policies, axis=-2, size=self.num_actions)

    return USFAPreds(
      sf=sfs,       # [N, A, D_w]
      policy=policies,         # [N, A, D_w]
      q_values=q_values,  # [N, A]
      task=task)         # [D_w]

class UsfaArch(hk.RNNCore):
  """Universal Successor Feature Approximator."""

  def __init__(self,
               torso: networks.OarTorso,
               memory: hk.RNNCore,
               head: SfGpiHead,
               name: str = 'usfa_arch'):
    super().__init__(name=name)
    self._torso = torso
    self._memory = memory
    self._head = head

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [D]
      state: hk.LSTMState,  # [D]
      evaluation: bool = False,
  ) -> Tuple[USFAPreds, hk.LSTMState]:
    torso_outputs = self._torso(inputs)  # [D+A+1]
    memory_input = jnp.concatenate(
      (torso_outputs.image, torso_outputs.action), axis=-1)

    core_outputs, new_state = self._memory(memory_input, state)

    # inputs = jax.tree_map(lambda x:x.astype(jnp.float32), inputs)
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
    return predictions, new_state

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> hk.LSTMState:
    return self._memory.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[USFAPreds, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    torso_outputs = hk.BatchApply(self._torso)(inputs)  # [T, B, D+A+1]

    memory_input = jnp.concatenate(
      (torso_outputs.image, torso_outputs.action), axis=-1)

    core_outputs, new_states = hk.static_unroll(
      self._memory, memory_input, state)

    # treat T,B like this don't exist with vmap
    predictions = jax.vmap(jax.vmap(self._head))(
        core_outputs,                # [T, B, D]
        inputs.observation['task'],  # [T, B]
      )
    return predictions, new_states

def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config) -> networks_lib.UnrollableNetwork:
  """Builds default USFA networks for Minigrid games."""

  num_actions = env_spec.actions.num_values
  state_features_dim = env_spec.observations.observation['state_features'].shape[0]

  def make_core_module() -> UsfaArch:
    vision_torso = networks.BabyAIVisionTorso(
        flatten=True,
        conv_dim=config.final_conv_dim,
        out_dim=config.conv_flat_dim)

    observation_fn = networks.OarTorso(
      num_actions=num_actions,
      vision_torso=vision_torso,
      output_fn=networks.TorsoOutput,
    )

    if config.head == 'independent':
      SfNetCls = IndependentSfHead
    elif config.head == 'monolithic':
      SfNetCls = MonolithicSfHead
    else:
      raise NotImplementedError

    sf_net = SfNetCls(
      layers=config.sf_layers,
      state_features_dim=state_features_dim,
      num_actions=num_actions,
      policy_layers=config.policy_layers,
      # combine_policy=config.combine_policy,
      )
    

    usfa_head = SfGpiHead(
      num_actions=num_actions,
      nsamples=config.nsamples,
      variance=config.variance,
      sf_net=sf_net,
      eval_task_support=config.eval_task_support)

    return UsfaArch(
      torso=observation_fn,
      memory=hk.LSTM(config.state_dim),
      head=usfa_head)

  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)
