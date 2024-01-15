
class ObjectOrientedUsfaArch(hk.RNNCore):
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
  def make_object_oriented_usfa_inputs(
      self,
      inputs: observation_action_reward.OAR,
      core_outputs: jnp.ndarray,
      ):
    action_embdder = lambda x: hk.Linear(128, w_init=hk.initializers.RandomNormal)(x)
    object_embdder = lambda x: hk.Linear(128, w_init=hk.initializers.RandomNormal)(x)

    # each are [B, N, D] where N differs for action and object embeddings
    action_embeddings = hk.BatchApply(action_embdder)(inputs.observation['actions'])
    object_embeddings = hk.BatchApply(object_embdder)(inputs.observation['objects'])
    option_inputs = jnp.concatenate((action_embeddings, object_embeddings), axis=-2)

    # vmap concat over middle dimension to replicate concat across all "actions"
    # [B, D1] + [B, A, D2] --> [B, A, D1+D2]
    concat = lambda a, b: jnp.concatenate((a,b), axis=-1)
    concat = jax.vmap(in_axes=(None, 1), out_axes=1)(concat)

    return concat(core_outputs, option_inputs)

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState,  # [B, ...]
      evaluation: bool = False,
  ) -> Tuple[USFAPreds, hk.LSTMState]:
    torso_outputs = self._torso(inputs)  # [B, D+A+1]
    memory_input = jnp.concatenate(
      (torso_outputs.image, torso_outputs.action), axis=-1)
    core_outputs, new_state = self._memory(memory_input, state)

    import ipdb; ipdb.set_trace()
    head_inputs = USFAInputs(
      task=inputs.observation['task'],
      usfa_input=self.make_object_oriented_usfa_inputs(inputs, core_outputs),
      train_tasks=inputs.observation['train_tasks'],
    )
    if evaluation:
      predictions = self._head.evaluate(head_inputs)
      import ipdb; ipdb.set_trace()
    else:
      predictions = self._head(head_inputs)
      import ipdb; ipdb.set_trace()
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

    head_inputs = USFAInputs(
      task=torso_outputs.task,
      usfa_input=core_outputs,
    )
    predictions = hk.BatchApply(self._head)(head_inputs)  # [T, B, A]
    return predictions, new_states

def make_object_oriented_minigrid_networks(
        env_spec: specs.EnvironmentSpec,
        config: Config) -> networks_lib.UnrollableNetwork:
  """Builds default USFA networks for Minigrid games."""

  num_actions = env_spec.actions.num_values
  state_features_dim = env_spec.observations.observation['state_features'].shape[0]

  def make_core_module() -> ObjectOrientedUsfaArch:
    vision_torso = networks.AtariVisionTorso()

    observation_fn = networks.OarTorso(
      num_actions=num_actions,
      vision_torso=vision_torso,
      output_fn=networks.TorsoOutput,
    )

    usfa_head = SfGpiHead(
      num_actions=num_actions,
      state_features_dim=state_features_dim,
      nsamples=config.nsamples,
      eval_task_support=config.eval_task_support)

    return ObjectOrientedUsfaArch(
      torso=observation_fn,
      memory=hk.LSTM(config.state_dim),
      head=usfa_head)

  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)
