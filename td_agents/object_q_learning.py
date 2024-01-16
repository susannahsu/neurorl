
from typing import Optional, Tuple, Optional, Callable, NamedTuple

import functools

from acme import specs
from acme.agents.jax import r2d2
from acme.jax.networks import base
from acme.jax import networks as networks_lib
from acme.wrappers import observation_action_reward
from acme.jax.networks import duelling
from acme import types as acme_types
from acme.agents.jax import actor_core as actor_core_lib

import dataclasses
import haiku as hk
from haiku._src.recurrent import LSTMState
import jax
import jax.numpy as jnp
import numpy as np
import rlax

import library.networks as networks

from library import utils 
from td_agents import basics
from td_agents import q_learning

Array = acme_types.NestedArray

@dataclasses.dataclass
class Config(q_learning.Config):
  q_dim: int = 512

class ObjectLSTMState(NamedTuple):
  hidden: jax.Array
  cell: jax.Array
  objects: jax.Array

def update_state(state: hk.LSTMState, objects: jax.Array):
  return ObjectLSTMState(
    hidden=state.hidden,
    cell=state.cell,
    objects=objects,
  )

class ObjectOrientedR2D2(hk.RNNCore):
  """A duelling recurrent network for use with Atari observations as seen in R2D2.

  See https://openreview.net/forum?id=r1lyTjAqYX for more information.
  """

  def __init__(self,
               vision_fn: networks.OarTorso,
               memory: hk.RNNCore,
               object_qhead: hk.Module,
               objects_spec: Tuple[int],
               task_encoder: Optional[hk.Module] = None,
               name: str = 'r2d2_arch'):
    super().__init__(name=name)
    self._vision_fn = vision_fn
    self._memory = memory
    self._object_qhead = object_qhead
    self._objects_spec = objects_spec
    self._task_encoder = task_encoder

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> hk.LSTMState:
    state = self._memory.initial_state(batch_size)
    return update_state(state, objects=jnp.zeros(self._objects_spec))

  def process_inputs(self,
                     inputs: observation_action_reward.OAR, 
                     prev_state: ObjectLSTMState):
    # convert action to onehot
    # prev_objects = prev_state.objects
    # prev_action = inputs.action
    num_primitive_actions = inputs.observation['actions'].shape[-1]

    # [A, A] array
    actions = jax.nn.one_hot(
        inputs.observation['actions'],
        num_classes=num_primitive_actions)

    # convert everything to floats
    inputs = jax.tree_map(lambda x: x.astype(jnp.float32), inputs)

    # objects_mask = inputs.observation['objects_mask']
    #----------------------------
    # process actions
    #----------------------------
    # 1 -> A+1: primitive actions
    # rest: object 
    # each are [N, D] where N differs for action and object embeddings
    def mlp(x):
      x = hk.Linear(128, w_init=hk.initializers.TruncatedNormal())(x)
      x = hk.Linear(128)(jax.nn.relu(x))
      return x
    action_mlp = hk.to_module(mlp)('action_mlp')
    object_mlp = hk.to_module(mlp)('object_mlp')

    actions = action_mlp(actions)
    objects = object_mlp(inputs.observation['objects'])

    #----------------------------
    # compute selected action
    #----------------------------
    # prev_objects = object_mlp(prev_objects)
    # all_actions = jnp.concatenate((actions, prev_objects), axis=-2)
    # chosen_action = all_actions[prev_action]

    #----------------------------
    # image + task + reward
    #----------------------------
    image = inputs.observation['image']/255.0
    image = self._vision_fn(image)  # [D+A+1]

    # task
    task = self._task_encoder(inputs.observation['task'])

    # reward = jnp.tanh(inputs.reward)
    reward = jnp.expand_dims(inputs.reward, axis=-1)

    state_input = jnp.concatenate(
      (image, task, reward), axis=-1)

    return state_input, objects

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:

    state_input, objects = self.process_inputs(inputs, state)
    core_outputs, new_state = self._memory(state_input, state)
    new_state = update_state(new_state, inputs.observation['objects'].astype(jnp.float32))
    q_values = self._object_qhead(core_outputs, objects)
    return q_values, new_state

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""

    state_input, objects = hk.BatchApply(
      jax.vmap(self.process_inputs))(inputs, state)  # [T, B, D+A+1]

    core_outputs, new_states = hk.static_unroll(
      self._memory, state_input, state)

    new_state = update_state(new_state, inputs.observation['objects'].astype(jnp.float32))

    q_values = hk.BatchApply(jax.vmap(self._head))(core_outputs, objects)  # [T, B, A]
    return q_values, new_states

def make_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config,
        task_encoder: Callable[[Array], Array] = lambda obs: None
        ) -> r2d2.R2D2Networks:
  """Builds default R2D2 networks for Atari games."""

  num_actions = env_spec.actions.num_values
  objects_spec = env_spec.observations.observation['objects'].shape
  def make_core_module() -> ObjectOrientedR2D2:

    def object_qhead(state: jax.Array, objects: jax.Array):
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

      qs = jnp.concatenate((primitive_qs, object_qs))

      return qs

    object_qhead = hk.to_module(object_qhead)("object_qhead")


    return ObjectOrientedR2D2(
      vision_fn=networks.BabyAIVisionTorso(
          conv_dim=0, out_dim=config.state_dim),
      memory=hk.LSTM(config.state_dim),
      objects_spec=objects_spec,
      task_encoder=task_encoder,
      object_qhead=object_qhead,
      )


  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)

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
      q_values = jnp.where(valid_actions, q_values, -jnp.inf)
      return jnp.argmax(q_values)

    # Using jax.lax.cond for conditional execution
    rng, q_rng = jax.random.split(rng)
    # Generate a random number to decide explore or exploit
    explore = jax.random.uniform(q_rng) < state.epsilon
    rng, q_rng = jax.random.split(rng)
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
