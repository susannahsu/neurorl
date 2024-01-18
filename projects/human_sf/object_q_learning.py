
import functools

from typing import Optional, Tuple, Optional, Callable


from acme import specs
from acme.agents.jax import r2d2
from acme.jax.networks import base
from acme.jax import networks as networks_lib
from acme.wrappers import observation_action_reward
from acme import types as acme_types
from acme.agents.jax import actor_core as actor_core_lib

import dataclasses
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import library.networks as networks

from td_agents import basics
from projects.human_sf import utils

Array = acme_types.NestedArray

@dataclasses.dataclass
class Config(basics.Config):
  q_dim: int = 512

class ObjectOrientedR2D2(hk.RNNCore):
  """A duelling recurrent network for use with Atari observations as seen in R2D2.

  See https://openreview.net/forum?id=r1lyTjAqYX for more information.
  """

  def __init__(self,
               vision_fn: networks.OarTorso,
               memory: hk.RNNCore,
               object_qhead: hk.Module,
               task_encoder: Optional[hk.Module] = None,
               name: str = 'r2d2_arch'):
    super().__init__(name=name)
    self._vision_fn = vision_fn
    self._memory = memory
    self._object_qhead = object_qhead
    self._task_encoder = task_encoder
    self.process_inputs = functools.partial(
      utils.process_inputs,
      vision_fn=vision_fn,
      task_encoder=task_encoder,
      )

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> hk.LSTMState:
    return self._memory.initial_state(batch_size)

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:

    state_input, objects, objects_mask = self.process_inputs(inputs)
    core_outputs, new_state = self._memory(state_input, state)

    q_values = self._object_qhead(core_outputs, objects, objects_mask)

    return q_values, new_state

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""

    state_input, objects, objects_mask = hk.BatchApply(
      jax.vmap(self.process_inputs))(inputs)  # [T, B, D+A+1]

    core_outputs, new_states = hk.static_unroll(
      self._memory, state_input, state)

    q_values = hk.BatchApply(jax.vmap(self._object_qhead))(core_outputs, objects, objects_mask)  # [T, B, A]
    return q_values, new_states

def get_actor_core(
    networks: basics.NetworkFn,
    config: Config,
    evaluation: bool = False,
    extract_q_values = lambda preds: preds
):
  """Returns ActorCore."""
  def select_action(params: networks_lib.Params,
                    observation: networks_lib.Observation,
                    state: basics.ActorState[actor_core_lib.RecurrentState]):
    rng, policy_rng = jax.random.split(state.rng)

    preds, recurrent_state = networks.apply(params, policy_rng, observation, state.recurrent_state)

    q_values = extract_q_values(preds)

    valid_actions = jnp.concatenate((
      jnp.ones(observation.observation['actions'].shape),
      observation.observation['objects_mask'].astype(jnp.float32)),
      axis=-1)

    def uniform(rng):
      logits = jnp.where(valid_actions, valid_actions, -jnp.inf)

      return jax.random.categorical(rng, logits)

    def exploit(rng):
      return jnp.argmax(q_values)

    # Using jax.lax.cond for conditional execution
    rng, q_rng = jax.random.split(rng)
    # Generate a random number to decide explore or exploit
    explore = jax.random.uniform(q_rng) < state.epsilon
    action = jax.lax.cond(explore, uniform, exploit, q_rng)

    return action, basics.ActorState(
        rng=rng,
        epsilon=state.epsilon,
        step=state.step + 1,
        predictions=preds,
        recurrent_state=recurrent_state,
        prev_recurrent_state=state.recurrent_state)

  def init(
      rng: networks_lib.PRNGKey
  ) -> basics.ActorState[actor_core_lib.RecurrentState]:
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
    return basics.ActorState(
        rng=rng,
        epsilon=epsilon,
        step=0,
        recurrent_state=initial_core_state,
        prev_recurrent_state=initial_core_state)

  def get_extras(
      state: basics.ActorState[actor_core_lib.RecurrentState]
  ) -> actor_core_lib.Extras:
    return {'core_state': state.prev_recurrent_state}

  return actor_core_lib.ActorCore(
    init=init,
    select_action=select_action,
    get_extras=get_extras)

def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config,
        task_encoder: Callable[[Array], Array] = lambda obs: None
        ) -> r2d2.R2D2Networks:
  """Builds default R2D2 networks for Atari games."""

  num_actions = env_spec.actions.num_values
  def make_core_module() -> ObjectOrientedR2D2:

    def object_qhead(state: jax.Array,
                     objects: jax.Array,
                     objects_mask: jax.Array):
      #--------------
      # primitive actions
      #--------------
      primitive_qhead=hk.nets.MLP([config.q_dim, num_actions])
      primitive_qs = primitive_qhead(state)

      #--------------
      # objects actions
      #--------------
      object_qhead = hk.nets.MLP([config.q_dim, 1])

      # vmap concat over middle dimension to replicate concat across all "actions"
      # [D1] + [A, D2] --> [A, D1+D2]
      concat = lambda a, b: jnp.concatenate((a,b), axis=-1)
      concat = jax.vmap(concat, in_axes=(None, 0), out_axes=0)

      object_inputs = concat(state, objects)
      object_qs = object_qhead(object_inputs)
      object_qs = jnp.squeeze(object_qs, axis=-1)

      # whereover objects not available, but -inf
      # important for learning
      object_qs =  jnp.where(objects_mask, object_qs, -jnp.inf)
      qs = jnp.concatenate((primitive_qs, object_qs))

      return qs

    object_qhead = hk.to_module(object_qhead)("object_qhead")


    return ObjectOrientedR2D2(
      vision_fn=networks.BabyAIVisionTorso(
          conv_dim=16, out_dim=config.state_dim),
      memory=hk.LSTM(config.state_dim),
      task_encoder=task_encoder,
      object_qhead=object_qhead,
      )


  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)

