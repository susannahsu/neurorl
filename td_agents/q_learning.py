
from typing import Optional, Tuple, Iterator, Optional


from acme import core
from acme import specs
from acme.agents.jax import r2d2
from acme.agents.jax.r2d2 import learning as r2d2_learning
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.jax.networks import base
from acme.jax import networks as networks_lib
from acme.utils import counting
from acme.utils import loggers
from acme.wrappers import observation_action_reward
from acme.jax.networks import duelling

import dataclasses
import haiku as hk
import optax
import reverb

import lib.networks as networks
from td_agents import basics


@dataclasses.dataclass
class Config(basics.Config):
  q_dim: int = 512

class R2D2Builder(basics.Builder):

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: r2d2_networks.R2D2Networks,
      dataset: Iterator[r2d2_learning.R2D2ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec
    optimizer_chain = []
    if self._config.max_grad_norm:
      optimizer_chain.append(
        optax.clip_by_global_norm(self._config.max_grad_norm))
    optimizer_chain.append(
      optax.adam(self._config.learning_rate, eps=self._config.adam_eps)),
    # The learner updates the parameters (and initializes them).
    return r2d2_learning.R2D2Learner(
        networks=networks,
        batch_size=self._batch_size_per_device,
        random_key=random_key,
        burn_in_length=self._config.burn_in_length,
        discount=self._config.discount,
        importance_sampling_exponent=(
            self._config.importance_sampling_exponent),
        max_priority_weight=self._config.max_priority_weight,
        target_update_period=self._config.target_update_period,
        iterator=dataset,
        optimizer=optax.chain(*optimizer_chain),
        bootstrap_n=self._config.bootstrap_n,
        tx_pair=self._config.tx_pair,
        clip_rewards=self._config.clip_rewards,
        replay_client=replay_client,
        counter=counter,
        logger=logger_fn('learner'))

class R2D2Arch(hk.RNNCore):
  """A duelling recurrent network for use with Atari observations as seen in R2D2.

  See https://openreview.net/forum?id=r1lyTjAqYX for more information.
  """

  def __init__(self,
               torso: networks.OarTorso,
               memory: hk.RNNCore,
               head: hk.Module,
               name: str = 'r2d2_arch'):
    super().__init__(name=name)
    self._torso = torso
    self._memory = memory
    self._head = head

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    embeddings = self._torso(inputs)  # [B, D+A+1]
    core_outputs, new_state = self._memory(embeddings, state)
    q_values = self._head(core_outputs)
    return q_values, new_state

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> hk.LSTMState:
    return self._memory.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    embeddings = hk.BatchApply(self._torso)(inputs)  # [T, B, D+A+1]
    core_outputs, new_states = hk.static_unroll(
      self._memory, embeddings, state)
    q_values = hk.BatchApply(self._head)(core_outputs)  # [T, B, A]
    return q_values, new_states

def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config,
        use_language: bool = False,
        ) -> r2d2.R2D2Networks:
  """Builds default R2D2 networks for Atari games."""

  num_actions = env_spec.actions.num_values

  def make_core_module() -> R2D2Arch:
    vision_torso = networks.AtariVisionTorso(
      out_dim=config.state_dim)

    if use_language:
      task_encoder = networks.LanguageEncoder(
          vocab_size=1000, word_dim=128)
    else:
      task_encoder = lambda obs: None

    observation_fn = networks.OarTorso(
      num_actions=num_actions,
      vision_torso=vision_torso,
      task_encoder=task_encoder,
    )

    return R2D2Arch(
      torso=observation_fn,
      memory=hk.LSTM(config.state_dim),
      head=duelling.DuellingMLP(num_actions,
                                hidden_sizes=[config.q_dim]))

  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)

