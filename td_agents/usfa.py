
from typing import Callable, NamedTuple, Tuple, Optional

import dataclasses
import functools
import jax
import jax.numpy as jnp
import haiku as hk
import rlax
import numpy as np

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

@dataclasses.dataclass
class Config(basics.Config):
  eval_task_support: str = "train"  # options:
  nsamples: int = 0  # no samples outside of train vector
  variance: float = 0.1

  final_conv_dim: int = 16
  conv_flat_dim: Optional[int] = 0
  sf_layers : Tuple[int]=(128, 128)
  policy_layers : Tuple[int]=(32,)

  sf_coeff: float = 1.0
  q_coeff: float = 0.0

def cumulants_from_env(data, online_preds, online_state, target_preds, target_state):
  # [T, B, C]
  cumulants = data.observation.observation['state_features']

  # ASSUMPTION: s' has cumulants for (s,a, r)
  cumulants = cumulants[1:]
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

@dataclasses.dataclass
class UsfaLossFn(basics.RecurrentLossFn):

  extract_cumulants: Callable = cumulants_from_env
  extract_task: Callable = lambda data: data.observation.observation['task']
  extract_task_dim: Callable = lambda x: x[:, :, 0]

  sf_coeff: float = 1.0
  q_coeff: float = 0.0

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
    td_error_fn = functools.partial(
            rlax.transformed_n_step_q_learning,
            n=self.bootstrap_n)

    # vmap over batch dimension (B), return B in dim=1
    td_error_fn = jax.vmap(td_error_fn, in_axes=1, out_axes=1)

    # vmap over policy dimension (N), return N in dim=2
    td_error_fn = jax.vmap(td_error_fn, in_axes=(2, None, 2, 2, None, None), out_axes=2)

    # vmap over cumulant dimension (C), return in dim=3
    td_error_fn = jax.vmap(td_error_fn, in_axes=(4, None, 4, None, 2, None), out_axes=3)

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

    metrics = {
      '0.total_loss': batch_loss,
      '0.1.sf_loss': sf_loss,
      '0.1.q_loss': q_loss,
      '2.q_td_error': jnp.abs(value_td_error),
      '2.sf_td_error': jnp.abs(sf_td_error),
      '3.cumulants': cumulants,
      '3.sf_mean': online_sf,
      '3.sf_var': online_sf.var(),
      }

    return value_td_error, batch_loss, metrics # [T, B], [B]

class USFAPreds(NamedTuple):
  q_values: jnp.ndarray  # q-value
  sf: jnp.ndarray # successor features
  policy: jnp.ndarray  # policy vector
  task: jnp.ndarray  # task vector (potentially embedded)

class MonolithicSfHead(hk.Module):
  def __init__(self, layers: int, num_actions: int, state_features_dim: int,
               name: Optional[str] = None):
    super(MonolithicSfHead, self).__init__(name=name)
    self.net = hk.nets.MLP(
        tuple(layers)+(num_actions * state_features_dim,))
    self.num_actions = num_actions
    self.state_features_dim = state_features_dim

  def __call__(self,
               sf_input: jnp.ndarray,
               policy: jnp.ndarray,
               task: jnp.ndarray) -> jnp.ndarray:
    """Compute successor features and q-valuesu
    
    Args:
        inputs (jnp.ndarray): D_1
        policy (jnp.ndarray): D_2
        task (jnp.ndarray): D_1
    
    Returns:
        jnp.ndarray: 2-D tensor of action values of shape [batch_size, num_actions]
    """
    sf_input = jnp.concatenate((sf_input, policy))  # 2D

    # [A * C]
    sf = self.net(sf_input)
    # [A, C]
    sf = jnp.reshape(sf, (self.num_actions, self.state_features_dim))

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
    state_features_dim: int,
    sf_net : hk.Module,
    policy_layers : Tuple[int]=(32,),
    nsamples: int=10,
    variance: Optional[float]=0.5,
    eval_task_support: str = 'train', 
    **kwargs,
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
    self.state_features_dim = state_features_dim
    self.var = variance
    self.nsamples = nsamples
    self.eval_task_support = eval_task_support

    if policy_layers:
      self.policy_net = hk.nets.MLP(policy_layers)
    else:
      self.policy_net = lambda x: x

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
      task=policy)

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

    policy_embeddings = self.policy_net(policies)
    sfs, q_values = jax.vmap(
      self.sf_net, in_axes=(None, 0, None), out_axes=0)(
        usfa_input,
        policy_embeddings,
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

    if evaluation:
      predictions = self._head.evaluate(
        task=inputs.observation['task'],
        usfa_input=core_outputs,
        train_tasks=inputs.observation['train_tasks']
        )
    else:
      predictions = self._head(
        task=inputs.observation['task'],
        usfa_input=core_outputs,
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
    predictions = jax.jit(hk.BatchApply(jax.vmap(self._head)))(
        task=inputs.observation['task'],  # [T, B, N]
        usfa_input=core_outputs,  # [T, B, D]
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

    sf_net = MonolithicSfHead(
      layers=config.sf_layers,
      state_features_dim=state_features_dim,
      num_actions=num_actions)

    usfa_head = SfGpiHead(
      num_actions=num_actions,
      state_features_dim=state_features_dim,
      nsamples=config.nsamples,
      variance=config.variance,
      sf_net=sf_net,
      policy_layers=config.policy_layers,
      eval_task_support=config.eval_task_support)

    return UsfaArch(
      torso=observation_fn,
      memory=hk.LSTM(config.state_dim),
      head=usfa_head)

  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)
