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
  --wandb_project=imagination_debug \
  --search='qlearning'

# JAX does Just In Time (JIT) compilation. remove this.
JAX_DISABLE_JIT=1 python -m ipdb -c continue configs/biorl_trainer_2learners.py \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --use_wandb=False \
  --wandb_entity=wcarvalho92 \
  --wandb_project=imagination_debug \
  --search='initial' \


# DEBUGGING, parallel
python -m ipdb -c continue configs/biorl_trainer_2learners.py \
  --parallel='sbatch' \
  --debug_parallel=True \
  --run_distributed=False \
  --use_wandb=True \
  --wandb_entity=wcarvalho92 \
  --wandb_project=imagination_debug \
  --search='initial'


# launch jobs on slurm
python configs/biorl_trainer_2learners.py \
  --parallel='sbatch' \
  --run_distributed=True \
  --use_wandb=True \
  --partition=kempner \
  --account=kempner_fellows \
  --wandb_entity=wcarvalho92 \
  --wandb_project=imagination \
  --search='initial'

"""
import functools 
from typing import List, Sequence, Optional, Callable

import dataclasses
from absl import flags
from absl import app
from absl import logging
import os
import datetime

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
from acme.jax import utils
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

from lib.dm_env_wrappers import GymWrapper
import lib.env_wrappers as env_wrappers
import lib.experiment_builder as experiment_builder
import lib.parallel as parallel
import lib.utils as utils
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
class QlearningConfig(q_learning.Config):

  eval_every: int = 100
  num_eval_episodes: int = 10

  online_update_period: int = 100
  offline_update_period: int = 2000

@dataclasses.dataclass
class MuZeroConfig(muzero.Config):

  eval_every: int = 100
  num_eval_episodes: int = 10

  online_update_period: int = 100
  offline_update_period: int = 2000

def make_online_replay_tables(
    environment_spec: specs.EnvironmentSpec,
    sequence_length: int,
    batch_size: int,
    replay_table_name: str = 'online_table',
    ) -> List[reverb.Table]:
  """Creates reverb tables for the algorithm.
  
  Copies from PPO: https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/ppo/builder.py
  """

  signature = adders_reverb.SequenceAdder.signature(
      environment_spec, sequence_length=sequence_length)
  replay_tables = [
      reverb.Table.queue(
          name=replay_table_name,
          max_size=batch_size,
          signature=signature)
  ]
    # Disable blocking of inserts by tables' rate limiters, as this function
  # executes learning (sampling from the table) and data generation
  # (inserting into the table) sequentially from the same thread
  # which could result in blocked insert making the algorithm hang.
  return disable_insert_blocking(
      replay_tables)

def make_offline_replay_tables(
    config: basics.Config,
    step_spec,
    sequence_length: int,
    sequence_period: int,
    replay_table_name: str = 'offline_table',
) -> List[reverb.Table]:
  """Create tables to insert data into.
  
  sequence_length = burn_in_length + trace_length + 1. this determines the length of the trajectory sampled during replay.
  """
  if config.samples_per_insert:
    samples_per_insert_tolerance = (
        config.samples_per_insert_tolerance_rate *
        config.samples_per_insert)
    error_buffer = config.min_replay_size * samples_per_insert_tolerance
    limiter = reverb.rate_limiters.SampleToInsertRatio(
        min_size_to_sample=config.min_replay_size,
        samples_per_insert=config.samples_per_insert,
        error_buffer=error_buffer)
  else:
    limiter = reverb.rate_limiters.MinSize(config.min_replay_size)

  return [
      reverb.Table(
          name=replay_table_name,
          sampler=reverb.selectors.Prioritized(
              config.priority_exponent),
          remover=reverb.selectors.Fifo(),
          max_size=config.max_replay_size,
          rate_limiter=limiter,
          signature=sw.infer_signature(
              configs=_make_adder_config(step_spec, sequence_length,
                                          sequence_period),
              step_spec=step_spec))
  ]

def run_experiment(
    config: basics.Config,
    log_dir: str,
    online_learner_cls: basics.SGDLearner,
    offline_learner_cls: basics.SGDLearner,
    get_actor_core: Callable[
      [basics.NetworkFn, basics.Config, Evaluation]],
    environment_factory: types.EnvironmentFactory,
    network_factory: experiments.config.NetworkFactory,
    logger_factory: Optional[loggers.LoggerFactory] = None,
    environment_spec: Optional[specs.EnvironmentSpec] = None,
    observers: Sequence[EnvLoopObserver] = (),
    eval_every: int = 100,
    num_eval_episodes: int = 1):
  """Runs a simple, single-threaded training loop using the default evaluators.

  It targets simplicity of the code and so only the basic features are supported.

  Important variables:
  - online_update_period: how often data is added to online buffer and used to update.
  - offline_sequence_period: how often data is added to offline buffer. If less than
      offline_sequence_length, overlapping sequences are added. If equal to
      offline_sequence_length, sequences are exactly non-overlapping.
  - offline_sequence_length: length of training trajector. should want {1,2} for goal of this project
  Arguments:
    experiment: Definition and configuration of the agent to run.
    eval_every: After how many actor steps to perform evaluation.
    num_eval_episodes: How many evaluation episodes to execute at each
      evaluation step.

  Copied from: https://github.com/google-deepmind/acme/blob/master/acme/jax/experiments/run_experiment.py
  """

  import ipdb; ipdb.set_trace()
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
      environment_spec=environment_spec,
      evaluation=False)

  #################################################
  # Setup online replay buffer
  # largely copied from PPO builder: https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/ppo/builder.py
  # this will train 
  # - observation function (e.g. CNN)
  # - state-function (LSTM)
  # - Q-function
  # - transition-model
  #################################################
  online_replay_table_name = 'online_table'
  #--------------------------
  # this will be used to store data
  #--------------------------
  online_replay_tables, _ = make_online_replay_tables(
    environment_spec=environment_spec,
    sequence_length=config.online_update_period,
    batch_size=config.online_batch_size,
    replay_table_name=online_replay_table_name,
    )
  online_replay_server = reverb.Server(online_replay_tables, port=None)
  online_replay_client = reverb.Client(f'localhost:{online_replay_server.port}')

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
  online_dataset = utils.multi_device_put(iterable=online_dataset, devices=jax.local_devices())
  online_dataset = utils.prefetch(online_dataset, buffer_size=1)

  # --------------------------
  # this will be used to add data
  # --------------------------
  online_adder = adders_reverb.SequenceAdder(
      client=online_replay_client,
      priority_fns={online_replay_table_name: None},
      period=config.online_update_period - 1,
      sequence_length=config.online_update_period,
  )

  #################################################
  # Setup offline replay buffer
  # mainly copied from R2D2 builder:
  # https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/r2d2/builder.py
  #################################################
  offline_update_period = config.offline_update_period
  offline_sequence_period = config.sequence_period
  offline_sequence_length = config.sequence_length
  offline_replay_table_name='offline_table'
  # Create the replay server and grab its address.
  dummy_actor_state = policy.init(jax.random.PRNGKey(0))
  extras_spec = policy.get_extras(dummy_actor_state)
  step_spec = structured.create_step_spec(
      environment_spec=environment_spec, extras_spec=extras_spec)
  offline_replay_tables = make_offline_replay_tables(
    config=config,
    step_spec=step_spec,
    sequence_length=offline_sequence_length,
    offline_sequence_period=offline_sequence_period,
    replay_table_name=offline_replay_table_name,
  )
  offline_replay_server = reverb.Server(offline_replay_tables, port=None)
  offline_replay_client = reverb.Client(f'localhost:{offline_replay_server.port}')

  #--------------------------
  # this will be used to iterate through data
  #--------------------------
  batch_size_per_learner = config.offline_batch_size // jax.process_count()
  offline_dataset = datasets.make_reverb_dataset(
      table=config.replay_table_name,
      server_address=offline_replay_client.server_address,
      batch_size=config.offline_batch_size // jax.device_count(),
      num_parallel_calls=None,
      max_in_flight_samples_per_worker=2 * batch_size_per_learner,
      postprocess=_zero_pad(offline_sequence_length),
  )

  offline_dataset = utils.multi_device_put(
        offline_dataset.as_numpy_iterator(),
        devices=jax.local_devices(),
        split_fn=utils.keep_key_on_host)
  # We always use prefetch as it provides an iterator with an additional
  # 'ready' method.
  offline_dataset = utils.prefetch(offline_dataset, buffer_size=1)

  # --------------------------
  # this will be used to add data
  # --------------------------
  offline_adder = structured.StructuredAdder(
      client=offline_replay_client,
      max_in_flight_items=5,
      configs=_make_adder_config(step_spec, offline_sequence_length,
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
  # KEY: online_learner will use step_data while offline_learner will do step
  #
  #################################################
  """
    training_state = offline_learner._state

    data = next(offline_dataset)
    training_state, metrics = offline_learner.step_data(data, training_state)
    offline_learner.update_state(training_state)

    data = next(online_dataset)
    training_state, metrics = online_learner.step_data(data, training_state)
    offline_learner.update_state(training_state)

  """
  # -------------------
  # offline learner
  # -------------------
  learner_key, key = jax.random.split(key)
  offline_learner = offline_learner_cls(
    network=networks,
    data_iterator=offline_dataset,
    target_update_period=config.target_update_period,
    random_key=learner_key,
    replay_client=offline_replay_client,  # only necessary if using client. not necessary
    replay_table_name=offline_replay_table_name,
    # counter=counting.Counter(parent_counter, prefix='learner', time_delta=0.),
    # logger=logger_factory['learner'],  # optional
    )

  #-------------------
  # online learner. 
  #-------------------
  learner_key, key = jax.random.split(key)

  online_learner = basics.SGDLearner(
    network=networks,
    data_iterator=online_dataset,
    random_key=learner_key,
    # counter=counting.Counter(parent_counter, prefix='learner', time_delta=0.),
    # logger=logger_factory['learner'],  # optional
    )

  #################################################
  # Create checkpointer
  #################################################
  checkpointing = experiments.CheckpointingConfig(
      directory=log_dir,
      max_to_keep=5,
      add_uid=False,
      checkpoint_ttl_seconds=int(datetime.timedelta(days=30).total_seconds()))
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
  # Make train and eval actors
  #################################################
  variable_client = variable_utils.VariableClient(
      offline_learner,
      key='actor_variables',
      # how often to update actor with parameters
      update_period=config.online_update_period)

  actor_key, key = jax.random.split(key)
  actor = basics.BasicActor(
      actor=policy, 
      random_key=actor_key,
      variable_client=variable_client,
      adders=[online_adder, offline_adder],
      backend='cpu')
  eval_actor = basics.BasicActor(
      actor=get_actor_core(  # main difference is this
          networks=networks,
          environment_spec=environment_spec,
          evaluation=True),
      random_key=jax.random.PRNGKey(config.seed),
      variable_client=variable_client,
      backend='cpu')

  #################################################
  # Create the environment loops used for training.
  # NOTE: you can expand these/write the for loops manually. left here for future work.
  #################################################
  eval_loop = acme.EnvironmentLoop(
      environment,
      eval_actor,
      # counter=eval_counter,
      # logger=eval_logger,
      observers=observers)

  def maybe_update(step_count: int, period: int,
      training_state,
      learner,
      iterator,
      ):
    if step_count % period != 0:
      return training_state, {}
    data = next(iterator)
    training_state, metrics = learner.step_data(data, training_state)
    learner.update_state(training_state)
    return training_state, metrics

  training_state = offline_learner.get_state()
  episode_idx = 0
  step_count = 0
  while True:
    episode_idx += 1
    episode_steps: int = 0
    episode_return: float = 0.0
    episode_results = {}

    # initialize episode
    timestep = environment.reset()
    actor.observe_first(timestep)
    for observer in observers:
      observer.observe_first(environment, timestep)

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
        observer.observe(environment, timestep)
      training_state, metrics = maybe_update(
          step_count=step_count,
          period=config.online_update_period,
          training_state=training_state,
          learner=online_learner,
          iterator=online_dataset,
      )
      training_state, metrics = maybe_update(
          step_count=step_count,
          period=offline_update_period,
          training_state=training_state,
          learner=offline_learner,
          iterator=offline_dataset,
      )
      actor.update()
      checkpointer.save()

    # -------------------
    # episode is over. collect stats
    # -------------------
    episode_results.update({
        'episode_length': episode_steps,
        'episode_return': episode_return})
    for observer in observers:
      episode_results.update(observer.get_metrics())

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
    from td_agents import qlearning
    config = QlearningConfig(**config_kwargs)

    network_factory = functools.partial(
        qlearning.make_minigrid_networks, config=config)

    get_actor_core = basics.get_actor_core

    online_learner_cls = functools.partial(
      basics.SGDLearner,
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
      loss_fn=q_learning.R2D2LossFn(
          discount=config.discount,
          importance_sampling_exponent=config.importance_sampling_exponent,
          burn_in_length=config.burn_in_length,
          max_replay_size=config.max_replay_size,
          max_priority_weight=config.max_priority_weight,
          bootstrap_n=config.bootstrap_n,
    ))

  elif agent == 'muzero':
    from td_agents import muzero
    config = muzero.Config(**config_kwargs)

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
      optimizer_cnstr=muzero.muzero_optimizer_constr,
      loss_fn=muzero.MuZeroLossFn(
          discount=config.discount,
          importance_sampling_exponent=config.importance_sampling_exponent,
          burn_in_length=config.burn_in_length,
          max_replay_size=config.max_replay_size,
          max_priority_weight=config.max_priority_weight,
          bootstrap_n=config.bootstrap_n,
          discretizer=discretizer,
          mcts_policy=mcts_policy,
          simulation_steps=config.simulation_steps,
          reanalyze_ratio=config.reanalyze_ratio,
          root_policy_coef=config.root_policy_coef,
          root_value_coef=config.root_value_coef,
          model_policy_coef=config.model_policy_coef,
          model_value_coef=config.model_value_coef,
          model_reward_coef=config.model_reward_coef
          )
    )

    offline_learner_cls = functools.partial(
      basics.SGDLearner,
      optimizer_cnstr=muzero.muzero_optimizer_constr,
      loss_fn=muzero.MuZeroLossFn(
          discount=config.discount,
          importance_sampling_exponent=config.importance_sampling_exponent,
          burn_in_length=config.burn_in_length,
          max_replay_size=config.max_replay_size,
          max_priority_weight=config.max_priority_weight,
          bootstrap_n=config.bootstrap_n,
          discretizer=discretizer,
          mcts_policy=mcts_policy,
          simulation_steps=config.simulation_steps,
          reanalyze_ratio=config.reanalyze_ratio,
          root_policy_coef=config.root_policy_coef,
          root_value_coef=config.root_value_coef,
          model_policy_coef=config.model_policy_coef,
          model_value_coef=config.model_value_coef,
          model_reward_coef=config.model_reward_coef
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
