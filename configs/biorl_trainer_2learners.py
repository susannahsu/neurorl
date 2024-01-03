"""
Running experiments:
--------------------
# DEBUGGING, single stream
python -m ipdb -c continue configs/biorl_trainer_2learners.py \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --use_wandb=False \
  --wandb_entity=wcarvalho92 \
  --wandb_project=forage_debug \
  --search='muzero'

# JAX does Just In Time (JIT) compilation. remove this. useful for debugging.
JAX_DISABLE_JIT=1 python -m ipdb -c continue configs/biorl_trainer_2learners.py \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --use_wandb=False \
  --wandb_entity=wcarvalho92 \
  --wandb_project=forage_debug \
  --search='muzero'

# launch jobs on slurm
python configs/biorl_trainer_2learners.py \
  --parallel='sbatch' \
  --use_wandb=True \
  --partition=kempner \
  --account=kempner_fellows \
  --wandb_entity=wcarvalho92 \
  --wandb_project=forage \
  --search='muzero'

"""

import sys
import functools 
from typing import List, Sequence, Optional, Callable

import dataclasses
from absl import flags
from absl import app
from absl import logging
import os
import datetime
import time

import acme
from acme import specs
from acme import wrappers as acme_wrappers
from acme.adders import reverb as adders_reverb
from acme.jax import variable_utils
from acme.jax import types
from acme.datasets import reverb as datasets
from acme.adders.reverb import structured
from acme.agents.jax.r2d2 import actor as r2d2_actor
from acme.utils import loggers
from acme.utils.observers import EnvLoopObserver
from acme.agents.jax.r2d2.builder import _make_adder_config, _zero_pad
from acme.jax import experiments
from acme.jax import utils as jax_utils
from acme.tf import savers
from acme.utils import counting

import jax
import reverb
from reverb import structured_writer as sw
from ray import tune
import gymnasium
import dm_env
import minigrid

from td_agents import q_learning, muzero

from library.dm_env_wrappers import GymWrapper
import library.env_wrappers as env_wrappers
import library.experiment_builder as experiment_builder
import library.parallel as parallel
import library.utils as utils
from td_agents import basics
from td_agents.basics import disable_insert_blocking

flags.DEFINE_string('config_file', '', 'config file')
flags.DEFINE_string('search', 'default', 'which search to use.')
flags.DEFINE_string(
    'parallel', 'none', "none: run 1 experiment. sbatch: run many experiments with SBATCH. ray: run many experiments with say. use sbatch with SLUM or ray otherwise.")
flags.DEFINE_bool(
    'debug', False, 'If in debugging mode, only 1st config is run.')
flags.DEFINE_bool(
    'make_path', True, 'Create a path under `FLAGS>folder` for the experiment')

FLAGS = flags.FLAGS

Evaluation = bool

@dataclasses.dataclass
class TwoLearnerConfig:
  eval_every: int = 100
  num_eval_episodes: int = 10

  # online learner options
  learn_online: bool = True
  online_burn_in_length: int = 0
  online_batch_size: int = 1  # number of batch elements
  online_update_period: int = 20  # time-window
  num_online_updates: int = 1

  # offline learner options
  learn_offline: bool = True
  min_replay_size: int = 100  # only start sampling after this much
  offline_burn_in_length: int = 0
  offline_trace_length: int = 0 # number of learning steps
  offline_batch_size: int = 32
  offline_update_period: int = 2_000  # how often learner updated offline
  num_offline_updates: int = 100

  sequence_period: int = 40  # how often data is added to buffer
  trace_length: int = 1  # how often data is added to buffer


def run_experiment(
    config: basics.Config,
    log_dir: str,
    online_learner_cls: basics.SGDLearner,
    offline_learner_cls: basics.SGDLearner,
    get_actor_core: Callable[
      [basics.NetworkFn, basics.Config, Evaluation], basics.Policy],
    environment_factory: types.EnvironmentFactory,
    network_factory: experiments.config.NetworkFactory,
    logger_factory: Optional[loggers.LoggerFactory] = None,
    environment_spec: Optional[specs.EnvironmentSpec] = None,
    observers: Sequence[EnvLoopObserver] = (),
    eval_every: int = 100,
    num_eval_episodes: int = 1):
  """
  Runs an experiment with a given configuration, learning environment, and neural network setup.

  Arguments:
    config: Configuration object defining parameters for the experiment.
    log_dir: Directory path for logging the experiment's outputs.
    online_learner_cls: The class for the online learner, typically an instance of SGDLearner.
    offline_learner_cls: The class for the offline learner, typically an instance of SGDLearner.
    get_actor_core: A callable that returns a policy object, given network functions and configuration.
    environment_factory: A factory function to create the learning environment.
    network_factory: A factory function to create the neural network used in the experiment.
    logger_factory: (Optional) A factory function to create a logger for the experiment.
    environment_spec: (Optional) Specification of the environment, detailing the observation and action spaces, etc.
    observers: (Optional) A sequence of observers that watch and record various aspects of the environment loop.
    eval_every: (Optional) Frequency of evaluation within the training loop, in terms of actor steps.
    num_eval_episodes: (Optional) The number of evaluation episodes to perform at each evaluation step.

  The logic of this function is as follows:
  - make environment
  - make jax networks which define neural network operations
  - make a policy, which determines how agent selects actions
  - setup {offline, online} x {replay_table, iterator, adder}
    - the replay_table is a buffer which stores data
    - the iterator is used to sample data from this buffer
    - the adders are given to an "actor" which interfaced with environment and adds data to this buffer
    - setup offline and online learners
      - learners share their optimizer and parameters
    - create checkpointing object
    - create actors for both training and evaluation
    - then collect samples and learn!

  pseudo-code example of learning:
    training_state = offline_learner.initialize()
    for t times-steps:
      if offline_update_period:
        data = next(offline_dataset)
        training_state = offline_learner.step_data(data, training_state)
        offline_learner.set_state(training_state)

      if online_update_period:
        data = next(online_dataset)
        training_state = online_learner.step_data(data, training_state)
        offline_learner.set_state(training_state)

  Some important notes on learners:
  - online:
    - we assume a batch_size of 1 but this can be changed via config
    - the buffer window size corresponds to `online_update_period` which is how 
      often an agent is updated
  - offline:
    - here, we assume batches of length B but only 1 data-point corresponding to
      the initial state
    - this is updated every `offline_update_period`
    - this supports prioritized replay and requiring a minimimum size before sampling
      - both can be set via config values

  Copied from: https://github.com/google-deepmind/acme/blob/master/acme/jax/experiments/run_experiment.py
  """

  key = jax.random.PRNGKey(config.seed)

  # Create the environment and get its spec.
  environment = environment_factory(config.seed)
  environment_spec = environment_spec or specs.make_environment_spec(
      environment)

  # Create the networks and policy.
  # policy is uses to select actions
  networks = network_factory(environment_spec)
  policy = get_actor_core(
      networks=networks,
      config=config,
      evaluation=False)

  dummy_actor_state = policy.init(jax.random.PRNGKey(0))
  extras_spec = policy.get_extras(dummy_actor_state)

  assert config.learn_offline or config.learn_online
  #################################################
  # ONLINE BUFFER
  # -------------
  # largely copied from PPO builder: https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/ppo/builder.py
  # this will train
  # - observation function (e.g. CNN)
  # - state-function (e.g. LSTM)
  # - transition-model
  # - predictions (e.g. value function, reward function, etc.)
  #################################################
  online_replay_table_name = 'online_table'
  online_replay_client = None
  online_replay_tables = []
  if config.learn_online:
    #--------------------------
    # this will be used to store data
    #--------------------------
    # Copied from PPO: https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/ppo/builder.py
    # Create the replay server and grab its address.
    online_replay_tables = [
        reverb.Table.queue(
            name=online_replay_table_name,
            max_size=config.online_update_period,
            signature=adders_reverb.SequenceAdder.signature(
                environment_spec,
                extras_spec=extras_spec,
                sequence_length=config.online_update_period))
    ]
    online_replay_tables, _ = basics.disable_insert_blocking(online_replay_tables)

    online_replay_server = reverb.Server(online_replay_tables, port=None)
    online_replay_client = reverb.Client(f'localhost:{online_replay_server.port}')

    # --------------------------
    # this will be used to add data to online buffer
    # --------------------------
    length_between_online_updates = config.online_update_period - config.simulation_steps - 1
    # when we learn from the model, we're going to simulate K time-steps into the future.
    # for the FINAL K time-steps, they will not get losses from K time-steps into the future, since this will go over the batch.
    # this is most harmful for the LAST time-step, which will not have any learning from future time-steps. 
    # to account for this, we have the PERIOD with which data is added be K time-steps less than the length of batches (sequence_length)
    logging.info(f"Will update online every {length_between_online_updates} steps")
    online_adder = adders_reverb.SequenceAdder(
        client=online_replay_client,
        priority_fns={online_replay_table_name: None},
        period=length_between_online_updates,
        sequence_length=config.online_update_period,
    )

    #--------------------------
    # this will be used to iterate through data
    #--------------------------
    iterator_batch_size, ragged = divmod(config.online_batch_size,
                                          jax.device_count())
    if ragged:
      raise ValueError(
          'Learner batch size must be divisible by total number of devices!')

    # We don't use datasets.make_reverb_dataset() here to avoid interleaving
    # and prefetching, that doesn't work well with can_sample() check on update.
    online_dataset = reverb.TrajectoryDataset.from_table_signature(
          server_address=online_replay_client.server_address,
          table=online_replay_table_name,
          max_in_flight_samples_per_worker=(
              2 * config.online_batch_size // jax.process_count()
          ),
      )
    online_dataset = online_dataset.batch(iterator_batch_size, drop_remainder=True)
    online_dataset = online_dataset.as_numpy_iterator()
    online_dataset = jax_utils.multi_device_put(
        iterable=online_dataset, devices=jax.local_devices())
    online_dataset = jax_utils.prefetch(online_dataset, buffer_size=1)

  #################################################
  # OFFLINE BUFFER
  # -------------
  # mainly copied from R2D2 builder: https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/r2d2/builder.py
  # this will train
  # - policy/value function using model
  #################################################
  offline_replay_table_name = adders_reverb.DEFAULT_PRIORITY_TABLE
  offline_replay_client = None
  offline_replay_tables = []
  if config.learn_offline:
    # NOTE: priortiized replay table seems to require name below

    # spec for every step that will be added to buffer
    step_spec = structured.create_step_spec(
        environment_spec=environment_spec, extras_spec=extras_spec)

    offline_batch_length = config.offline_burn_in_length + \
        config.offline_trace_length + 1
    offline_sequence_period = offline_batch_length
    #--------------------------
    # this will be used to store data
    #--------------------------
    samples_per_insert=1.0
    samples_per_insert_tolerance = (
      config.samples_per_insert_tolerance_rate *
      samples_per_insert)
    error_buffer = config.min_replay_size * samples_per_insert_tolerance
    offset = samples_per_insert * config.min_replay_size
    min_diff = offset - error_buffer
    # max_diff = offset + error_buffer
    offline_replay_tables = [
      reverb.Table(
        name=offline_replay_table_name,
        sampler=reverb.selectors.Prioritized(config.priority_exponent),
        remover=reverb.selectors.Fifo(),
        max_size=config.max_replay_size,
        rate_limiter=reverb.rate_limiters.RateLimiter(
            samples_per_insert=samples_per_insert,
            min_size_to_sample=config.min_replay_size,
            min_diff=min_diff,
            max_diff=sys.float_info.max),
        signature=sw.infer_signature(
          configs=_make_adder_config(
              step_spec, offline_batch_length, offline_sequence_period),
          step_spec=step_spec))
  ]
    offline_replay_server = reverb.Server(offline_replay_tables, port=None)
    offline_replay_client = reverb.Client(f'localhost:{offline_replay_server.port}')

    #--------------------------
    # this will be used to iterate through data
    #--------------------------
    batch_size_per_learner = config.offline_batch_size // jax.process_count()
    offline_dataset = datasets.make_reverb_dataset(
        table=offline_replay_table_name,
        server_address=offline_replay_client.server_address,
        batch_size=config.offline_batch_size // jax.device_count(),
        num_parallel_calls=None,
        max_in_flight_samples_per_worker=2 * batch_size_per_learner,
        postprocess=_zero_pad(offline_batch_length),
    )

    offline_dataset = jax_utils.multi_device_put(
          offline_dataset.as_numpy_iterator(),
          devices=jax.local_devices(),
          split_fn=jax_utils.keep_key_on_host)
    # We always use prefetch as it provides an iterator with an additional
    # 'ready' method.
    offline_dataset = jax_utils.prefetch(offline_dataset, buffer_size=1)

    # --------------------------
    # this will be used to add data
    # --------------------------
    offline_adder = structured.StructuredAdder(
        client=offline_replay_client,
        max_in_flight_items=5,
        configs=_make_adder_config(step_spec, offline_batch_length,
                                  offline_sequence_period),
        step_spec=step_spec)

  #################################################
  # Make counter to track stats
  #################################################
  # Parent counter allows to share step counts between train and eval loops and
  # the learner, so that it is possible to plot for example evaluator's return
  # value as a function of the number of training episodes.
  parent_counter = counting.Counter(time_delta=0.)

  #################################################
  # Make learners
  #################################################
  learner_logger = logger_factory('learner')

  # -------------------
  # offline learner
  # -------------------
  learner_key, key = jax.random.split(key)
  
  offline_learner = offline_learner_cls(
    initialize=False,  # will manually init + share across learners
    network=networks,
    target_update_period=config.target_update_period,
    # below is for prioritizd replay
    replay_client=offline_replay_client,
    replay_table_name=offline_replay_table_name,
    # below is for tracking learning steps
    counter=counting.Counter(parent_counter, prefix='offline_learner', time_delta=0.),
    )


  #-------------------
  # online learner. 
  #-------------------
  online_learner = online_learner_cls(
    initialize=False,  # will manually init + share across learners
    network=networks,
    target_update_period=config.target_update_period,
    # below is for tracking learning steps
    counter=counting.Counter(parent_counter, prefix='online_learner', time_delta=0.),
    )
  
  #-------------------
  # initialize and share optimizer + training state
  #-------------------
  if config.learn_offline:
    training_state, optimizer = offline_learner.initialize(
      network=networks, 
      random_key=learner_key)
  else:
    training_state, optimizer = online_learner.initialize(
      network=networks,
      random_key=learner_key)

  offline_learner.set_optimizer(optimizer)
  online_learner.set_optimizer(optimizer)

  # NOTE: variable client is linked to a learner.
  # it will ask a learner for the state (which contains parameters) when updating the actor
  # therefore want to give the learners the training state.
  # give to both just to be safe. in reality, to whichever
  # the actor uses to get parameters from.
  offline_learner.set_state(training_state)
  online_learner.set_state(training_state)

  #################################################
  # Create checkpointer
  #################################################
  checkpointing = experiments.CheckpointingConfig(
      directory=log_dir,
      max_to_keep=5,
      add_uid=False,
      time_delta_minutes=60,  # save every 60 minutes
      checkpoint_ttl_seconds=int(
        datetime.timedelta(days=30).total_seconds())  # deleted after 30 days
      )
  checkpointer = savers.Checkpointer(
      objects_to_save={'online_learner': online_learner,
                        'offline_learner': offline_learner,
                        'counter': parent_counter},
      time_delta_minutes=checkpointing.time_delta_minutes,
      directory=checkpointing.directory,
      add_uid=checkpointing.add_uid,
      max_to_keep=checkpointing.max_to_keep,
      keep_checkpoint_every_n_hours=checkpointing.keep_checkpoint_every_n_hours,
      checkpoint_ttl_seconds=checkpointing.checkpoint_ttl_seconds,
  )


  #################################################
  # Make train and eval environment actors
  #################################################
  variable_client = variable_utils.VariableClient(
      offline_learner if config.learn_offline else online_learner,
      key='actor_variables',
      # how often to update actor with parameters
      update_period=length_between_online_updates if config.learn_online else config.offline_update_period)

  actor_adders = []
  if config.learn_offline:
    actor_adders.append(offline_adder)
  if config.learn_online:
    actor_adders.append(online_adder)

  actor_key, key = jax.random.split(key)
  actor = basics.BasicActor(
      actor=policy,
      random_key=actor_key,
      variable_client=variable_client,
      adders=actor_adders,
      backend='cpu')
  eval_actor = basics.BasicActor(
      actor=get_actor_core(networks=networks,
        config=config, evaluation=True),
      random_key=jax.random.PRNGKey(config.seed),
      variable_client=variable_client,
      backend='cpu')

  #################################################
  # Create the environment loops used for training.
  #################################################
  eval_counter = counting.Counter(
      parent_counter, prefix='evaluator', time_delta=0.)
  eval_loop = acme.EnvironmentLoop(
      environment,
      eval_actor,
      counter=eval_counter,
      logger=logger_factory('evaluator', eval_counter.get_steps_key(), 0),
      observers=observers)

  def maybe_update(
      step_count: int, period: int, num_updates: int,
      training_state,
      learner,
      table,
      iterator,
      logger,
      name: str = 'online',
      ):
    """If the iterator is ready, sample and do update.
    Otherwise check if table has enough data. if not, exit.

    basically copied from https://github.com/google-deepmind/acme/blob/1177501df180edadd9f125cf5ee960f74bff64af/acme/jax/experiments/run_experiment.py#L230
    """
    updating = step_count % period == 0
    steps = 0
    updates = 0
    trained = False
    if not updating:
      return training_state, trained
    while True:
      steps += 1
      if iterator.ready():
        data = next(iterator)
        training_state, metrics = learner.step_data(data, training_state)
        learner.set_state(training_state)
        updates += 1
        trained = True
        logger.write(
            {f'{name}_{k}': v for k, v in metrics.items()})
        if updates >= num_updates:
          return training_state, trained
      else:
        if not table.can_sample(1):
          return training_state, trained
        # Let iterator's prefetching thread get data from the table(s).
        time.sleep(0.001)

  episode_idx = 0
  step_count = 0
  actor_counter = counting.Counter(
      parent_counter, prefix='actor', time_delta=0.)
  actor_logger = logger_factory('actor', actor_counter.get_steps_key(), 0)
  while True:
    episode_idx += 1
    episode_steps: int = 0
    episode_return: float = 0.0
    episode_metrics = {}

    # initialize episode
    timestep = environment.reset()
    actor.observe_first(timestep)
    for observer in observers:
      observer.observe_first(environment, timestep)

    step_count += 1

    #-------------------
    # run episode
    # -------------------
    while not timestep.last():
      action = actor.select_action(timestep.observation)
      timestep = environment.step(action)

      step_count += 1
      episode_steps += 1
      episode_return += timestep.reward

      actor.observe(action, next_timestep=timestep)
      for observer in observers:
        observer.observe(environment, timestep, action)

      trained = False
      if config.learn_online:
        training_state, trained = maybe_update(
            step_count=step_count,
            num_updates=config.num_online_updates,
            period=config.online_update_period,
            training_state=training_state,
            table=online_replay_tables[0],
            learner=online_learner,
            iterator=online_dataset,
            logger=learner_logger,
            name="online",
        )
      if config.learn_offline:
        training_state, trained = maybe_update(
            step_count=step_count,
            num_updates=config.num_offline_updates,
            period=config.offline_update_period,
            training_state=training_state,
            table=offline_replay_tables[0],
            learner=offline_learner,
            iterator=offline_dataset,
            logger=learner_logger,
            name="offline",
        )
      if trained:
        actor.update()
        checkpointer.save()

    # -------------------
    # episode is over. collect stats
    # -------------------
    episode_metrics.update({
        'episode_length': episode_steps,
        'episode_return': episode_return})

    for observer in observers:
      episode_metrics.update(observer.get_metrics())

    # update metrics with episode count statistics
    counts = actor_counter.increment(episodes=1, steps=episode_steps)
    episode_metrics.update(counts)
    actor_logger.write(episode_metrics)

    # -------------------
    # do eval and exit if over limit
    # -------------------
    if num_eval_episodes and episode_idx % eval_every == 0:
      eval_loop.run(num_episodes=num_eval_episodes)
    if step_count > config.num_steps:
      break

  environment.close()

def make_environment(seed: int,
                     level="BabyAI-GoToRedBallNoDists-v0",
                     evaluation: bool = False,
                     **kwargs) -> dm_env.Environment:
  """Loads environments. For now, just "goto X" from minigrid. change as needed.
  
  Args:
      evaluation (bool, optional): whether evaluation.
  
  Returns:
      dm_env.Environment: Multitask environment is returned.
  """
  del seed
  del evaluation

  # create gymnasium.Gym environment
  # environments: https://minigrid.farama.org/environments/babyai/
  env = gymnasium.make(level)
  env = minigrid.wrappers.RGBImgPartialObsWrapper(env)
  env = env_wrappers.DictObservationSpaceWrapper(env)

  # convert to dm_env.Environment enironment
  env = GymWrapper(env)

  ####################################
  # ACME wrappers
  ####################################
  # add acme wrappers
  wrapper_list = [
    # put action + reward in observation
    acme_wrappers.ObservationActionRewardWrapper,
    # cheaper to do computation in single precision
    acme_wrappers.SinglePrecisionWrapper,
  ]

  return acme_wrappers.wrap_all(env, wrapper_list)

def train_single(
    env_kwargs: dict = None,
    wandb_init_kwargs: dict = None,
    agent_config_kwargs: dict = None,
    log_dir: str = None,
    num_actors: int = 1,
    run_distributed: bool = False,
):
  del num_actors

  config_kwargs = agent_config_kwargs or dict()
  env_kwargs = env_kwargs or dict()

  # -----------------------
  # load agent config, builder, network factory
  # -----------------------
  agent = agent_config_kwargs.get('agent', '')
  assert agent != '', 'please set agent'

  if agent == 'qlearning':
    config = utils.merge_configs(
        dataclass_configs=[q_learning.Config(), TwoLearnerConfig()],
        dict_configs=config_kwargs,
    )

    network_factory = functools.partial(
        q_learning.make_minigrid_networks, config=config)

    get_actor_core = basics.get_actor_core

    online_learner_cls = functools.partial(
      basics.SGDLearner,
      optimizer_cnstr=functools.partial(
          basics.default_adam_constr, config=config),
      loss_fn=q_learning.R2D2LossFn(
          discount=config.discount,
          importance_sampling_exponent=config.importance_sampling_exponent,
          burn_in_length=config.burn_in_length,
          max_replay_size=config.max_replay_size,
          max_priority_weight=config.max_priority_weight,
          bootstrap_n=config.bootstrap_n,
    ))

    offline_learner_cls = functools.partial(
      basics.SGDLearner,
      optimizer_cnstr=functools.partial(
          basics.default_adam_constr, config=config),
      loss_fn=q_learning.R2D2LossFn(
          discount=config.discount,
          importance_sampling_exponent=config.importance_sampling_exponent,
          burn_in_length=config.burn_in_length,
          max_replay_size=config.max_replay_size,
          max_priority_weight=config.max_priority_weight,
          bootstrap_n=1, # 1 time-step
    ))

  elif agent == 'muzero':
    from td_agents import muzero
    config = utils.merge_configs(
        dataclass_configs=[muzero.Config(), TwoLearnerConfig()],
        dict_configs=config_kwargs,
    )

    import mctx
    # currently using same policy in learning and acting
    mcts_policy = functools.partial(
      mctx.gumbel_muzero_policy,
      max_depth=config.max_sim_depth,
      num_simulations=config.num_simulations,
      gumbel_scale=config.gumbel_scale)

    discretizer = utils.Discretizer(
                  num_bins=config.num_bins,
                  max_value=config.max_scalar_value,
                  tx_pair=config.tx_pair,
              )

    network_factory = functools.partial(
        muzero.make_minigrid_networks, config=config)

    get_actor_core = functools.partial(
        muzero.get_actor_core,
        mcts_policy=mcts_policy,
        discretizer=discretizer,
    )

    online_learner_cls = functools.partial(
      basics.SGDLearner,
      optimizer_cnstr=functools.partial(muzero.muzero_optimizer_constr, config=config),
      loss_fn=muzero.MuZeroLossFn(
          discount=config.discount,
          burn_in_length=config.online_burn_in_length,
          max_priority_weight=config.max_priority_weight,
          bootstrap_n=config.bootstrap_n,
          discretizer=discretizer,
          simulation_steps=config.simulation_steps,
          importance_sampling_exponent=0.0,  # online so no importance
          value_target_source='return',  # use reward return as learning target,
          # learn value predictions (both root and model), and reward predictions (from model)
          root_policy_coef=0.0,
          model_policy_coef=0.0,
          mcts_policy=None,  # will not use MCTS
          root_value_coef=config.root_value_coef,  # value predictions
          model_value_coef=config.model_value_coef,
          model_reward_coef=config.model_reward_coef
          )
    )

    offline_learner_cls = functools.partial(
      basics.SGDLearner,
      optimizer_cnstr=functools.partial(muzero.muzero_optimizer_constr, config=config),
      loss_fn=muzero.MuZeroLossFn(
          discount=config.discount,
          importance_sampling_exponent=config.importance_sampling_exponent,
          burn_in_length=config.burn_in_length,
          max_priority_weight=config.max_priority_weight,
          discretizer=discretizer,
          value_target_source='mcts',  # use mcts for value updates
          root_policy_coef=config.root_policy_coef,
          root_value_coef=config.root_value_coef,
          mcts_policy=mcts_policy,
          # only learn root policy and value predictions USING model
          # don't learn model components
          model_policy_coef=0.0,
          model_value_coef=0.0,
          model_reward_coef=0.0,
          bootstrap_n=0,  # will not boot-strap for returns, using MCTS
          simulation_steps=0,  # not learning model with simulation steps
          )
    )

  else:
    raise NotImplementedError(agent)

  # -----------------------
  # load environment factory
  # -----------------------
  environment_factory = functools.partial(
    make_environment,
    **env_kwargs)

  # -----------------------
  # setup observer factory for environment
  # this logs the average every reset=50 episodes (instead of every episode)
  # -----------------------
  observers = [
      utils.LevelAvgReturnObserver(
        reset=50,
        get_task_name=lambda e: "task"
        ),
      ]

  # -----------------------
  # setup logger factory
  # -----------------------
  logger_factory = experiment_builder.setup_logger_factory(
      agent_config=config,
      log_dir=log_dir,
      wandb_init_kwargs=wandb_init_kwargs,
      actor_label="actor",
      evaluator_label="evaluator",
      learner_label="learner",
  )

  if run_distributed:
    raise NotImplementedError('distributed not implemented')
  else:
    run_experiment(
      log_dir=log_dir,
      config=config,
      get_actor_core=get_actor_core,
      online_learner_cls=online_learner_cls,
      offline_learner_cls=offline_learner_cls,
      environment_factory=environment_factory,
      network_factory=network_factory,
      observers=observers,
      logger_factory=logger_factory,
      eval_every=config.eval_every,
      num_eval_episodes=config.num_eval_episodes)

def setup_wandb_init_kwargs():
  if not FLAGS.use_wandb:
    return dict()

  wandb_init_kwargs = dict(
      project=FLAGS.wandb_project,
      entity=FLAGS.wandb_entity,
      notes=FLAGS.wandb_notes,
      name=FLAGS.wandb_name,
      group=FLAGS.search,
      save_code=False,
  )
  return wandb_init_kwargs

def run_single():
  ########################
  # default settings
  ########################
  env_kwargs = dict()
  agent_config_kwargs = dict()
  num_actors = FLAGS.num_actors
  run_distributed = FLAGS.run_distributed
  wandb_init_kwargs = setup_wandb_init_kwargs()
  if FLAGS.debug:
    agent_config_kwargs.update(dict(
      online_update_period=5,  # update every 5 steps
      simulation_steps=1,
      offline_batch_size=2,
      offline_update_period=4,  # update every 4 steps
    ))
    env_kwargs.update(dict(
    ))

  folder = FLAGS.folder or os.environ.get('RL_RESULTS_DIR', None)
  if not folder:
    folder = '/tmp/rl_results'

  if FLAGS.make_path:
    # i.e. ${folder}/runs/${date_time}/
    folder = parallel.gen_log_dir(
        base_dir=os.path.join(folder, 'rl_results'),
        hourminute=True,
        date=True,
    )

  ########################
  # override with config settings, e.g. from parallel run
  ########################
  if FLAGS.config_file:
    configs = utils.load_config(FLAGS.config_file)
    config = configs[FLAGS.config_idx-1]  # starts at 1 with SLURM
    logging.info(f'loaded config: {str(config)}')

    agent_config_kwargs.update(config['agent_config'])
    env_kwargs.update(config['env_config'])
    folder = config['folder']

    num_actors = config['num_actors']
    run_distributed = config['run_distributed']

    wandb_init_kwargs['group'] = config['wandb_group']
    wandb_init_kwargs['name'] = config['wandb_name']
    wandb_init_kwargs['project'] = config['wandb_project']
    wandb_init_kwargs['entity'] = config['wandb_entity']

    if not config['use_wandb']:
      wandb_init_kwargs = dict()


  if FLAGS.debug and not FLAGS.subprocess:
      configs = parallel.get_all_configurations(spaces=sweep(FLAGS.search))
      first_agent_config, first_env_config = parallel.get_agent_env_configs(
          config=configs[0])
      agent_config_kwargs.update(first_agent_config)
      env_kwargs.update(first_env_config)

  train_single(
    wandb_init_kwargs=wandb_init_kwargs,
    env_kwargs=env_kwargs,
    agent_config_kwargs=agent_config_kwargs,
    log_dir=folder,
    num_actors=num_actors,
    run_distributed=run_distributed
    )

def run_many():
  wandb_init_kwargs = setup_wandb_init_kwargs()

  folder = FLAGS.folder or os.environ.get('RL_RESULTS_DIR', None)
  if not folder:
    folder = '/tmp/rl_results_dir'

  assert FLAGS.debug is False, 'only run debug if not running many things in parallel'

  if FLAGS.parallel == 'ray':
    parallel.run_ray(
      wandb_init_kwargs=wandb_init_kwargs,
      use_wandb=FLAGS.use_wandb,
      debug=FLAGS.debug,
      folder=folder,
      space=sweep(FLAGS.search),
      make_program_command=functools.partial(
        parallel.make_program_command,
        trainer_filename=__file__,
        run_distributed=FLAGS.run_distributed,
        num_actors=FLAGS.num_actors),
    )
  elif FLAGS.parallel == 'sbatch':
    parallel.run_sbatch(
      trainer_filename=__file__,
      wandb_init_kwargs=wandb_init_kwargs,
      use_wandb=FLAGS.use_wandb,
      folder=folder,
      run_distributed=FLAGS.run_distributed,
      search_name=FLAGS.search,
      debug=FLAGS.debug_parallel,
      spaces=sweep(FLAGS.search),
      num_actors=FLAGS.num_actors)

def sweep(search: str = 'default'):
  if search == 'qlearning':
    space = [
        {
            "agent": tune.grid_search(['qlearning']),
            "seed": tune.grid_search([1]),
            "env.level": tune.grid_search([
                "BabyAI-GoToRedBallNoDists-v0",
                "BabyAI-GoToObjS6-v1",
            ]),
        }
    ]
  elif search == 'muzero':
    space = [
        {
            "agent": tune.grid_search(['muzero']),
            "seed": tune.grid_search([1]),
            "group": tune.grid_search(['muzero-4']),
            "action_source": tune.grid_search(['value']),
            "offline_update_period": tune.grid_search([2_000]),
            "num_offline_updates": tune.grid_search([1, 10]),
            "env.level": tune.grid_search([
                "BabyAI-GoToRedBallNoDists-v0",
                # "BabyAI-GoToObjS6-v1",
            ]),
        }
    ]
  else:
    raise NotImplementedError(search)

  return space

def main(_):
  assert FLAGS.parallel in ('ray', 'sbatch', 'none')
  if FLAGS.parallel in ('ray', 'sbatch'):
    run_many()
  else:
    run_single()

if __name__ == '__main__':
  app.run(main)
