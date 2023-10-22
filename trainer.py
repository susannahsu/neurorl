
import functools 

from absl import flags
from absl import app
from ray import tune
from absl import logging
from launchpad.nodes.python.local_multi_processing import PythonProcess
import launchpad as lp

from acme import wrappers as acme_wrappers
from acme.jax import experiments
from acme.utils import paths
import dm_env
import minigrid

import envs
import env_wrappers
import experiment_builder
import experiment_logger
import parallel
import utils

import r2d2

flags.DEFINE_string('search', 'default', 'which search to use.')
flags.DEFINE_bool(
    'train_single', False, 'Run many or 1 experiments')
flags.DEFINE_bool(
    'make_path', False, 'Create a path under `FLAGS>folder` for the experiment')
flags.DEFINE_bool(
    'auto_name_wandb', False, 'automatically name wandb.')
FLAGS = flags.FLAGS


def make_environment(seed: int, evaluation: bool = False) -> dm_env.Environment:
  """Loads environments.
  
  Args:
      evaluation (bool, optional): whether evaluation.
  
  Returns:
      dm_env.Environment: Multitask environment is returned.
  """

  gym_wrappers = [
    minigrid.wrappers.DictObservationSpaceWrapper,
    env_wrappers.GotoOptionsWrapper,
    env_wrappers.PickupCategoryCumulantsWrapper,
    functools.partial(minigrid.wrappers.RGBImgObsWrapper,
                      tile_size=8),
  ]

  fixed_door_locs = False if evaluation else True
  env = envs.KeyRoom(
    num_dists=0,
    fixed_door_locs=fixed_door_locs)
  
  for w in gym_wrappers:
    env = w(env)

  wrapper_list = [
    acme_wrappers.ObservationActionRewardWrapper,
    acme_wrappers.SinglePrecisionWrapper,
  ]

  return acme_wrappers.wrap_all(env, wrapper_list)


def setup_agents(
    agent: str,
    config_kwargs: dict = None,
    env_kwargs: dict = None,
    debug: bool = False,
    update_logger_kwargs: dict = None,
    setup_kwargs: dict = None,
    config_class: r2d2.R2D2Config = None,
):
  config_kwargs = config_kwargs or dict()
  update_logger_kwargs = update_logger_kwargs or dict()
  setup_kwargs = setup_kwargs or dict()

  # -----------------------
  # load agent config, builder, network factory
  # -----------------------
  if agent == 'uvfa_flat':
    config_class = config_class or r2d2.R2D2Config
    config = config_class(**config_kwargs)
    builder = r2d2.R2D2Builder(config)
    network_factory = functools.partial(
            r2d2.make_minigrid_networks, config=config)

  else:
    raise NotImplementedError

  return config, builder, network_factory

def setup_experiment_inputs(
    agent : str,
    agent_config_kwargs: dict=None,
    agent_config_file: str=None,
    env_kwargs: dict=None,
    env_config_file: str=None,
    debug: bool = False,
  ):
  """Setup."""

  # -----------------------
  # load agent and environment kwargs (potentially from files)
  # -----------------------
  config_kwargs = agent_config_kwargs or dict()
  if agent_config_file:
    config_kwargs = utils.load_config(agent_config_file)
  logging.info(f'config_kwargs: {str(config_kwargs)}')

  env_kwargs = env_kwargs or dict()
  if env_config_file:
    env_kwargs = utils.load_config(env_config_file)
  logging.info(f'env_kwargs: {str(env_kwargs)}')

  # -----------------------
  # load agent config, builder, network factory
  # -----------------------
  # Configure the agent & update with config kwargs
  config, builder, network_factory = setup_agents(
      agent=agent,
      debug=debug,
      config_kwargs=config_kwargs,
      env_kwargs=env_kwargs,
      update_logger_kwargs=dict(
          action_names=['left', 'right', 'forward', 'pickup_1',
                        'pickup_2', 'place', 'toggle', 'slice'],
      )
  )

  # -----------------------
  # setup observer factory for environment
  # -----------------------
  observers = [
      utils.LevelAvgReturnObserver(
              # get_task_name=lambda env: str(env.env.current_levelname),
              reset=50 if not debug else 5),
      ]

  return experiment_builder.OnlineExperimentConfigInputs(
    agent_config=config,
    final_env_kwargs=env_kwargs,
    builder=builder,
    network_factory=network_factory,
    environment_factory=make_environment,
    observers=observers,
  )

def train_single(
    default_env_kwargs: dict = None,
    wandb_init_kwargs: dict = None,
    agent_config_kwargs: dict = None,
    **kwargs,
):

  debug = FLAGS.debug

  experiment_config_inputs = setup_experiment_inputs(
    agent=FLAGS.agent,
    path=FLAGS.path,
    agent_config_kwargs=agent_config_kwargs,
    agent_config_file=FLAGS.agent_config,
    env_kwargs=default_env_kwargs,
    env_config_file=FLAGS.env_config,
    debug=debug)

  log_dir = FLAGS.folder
  if FLAGS.make_path:
    log_dir = experiment_logger.gen_log_dir(
        base_dir=log_dir,
        hourminute=True,
        date=True,
    )
    if FLAGS.auto_name_wandb and wandb_init_kwargs is not None:
      date_time = experiment_logger.date_time(time=True)
      logging.info(f'wandb name: {str(date_time)}')
      wandb_init_kwargs['name'] = date_time

  tasks_file = experiment_config_inputs.final_env_kwargs['tasks_file']
  logger_factory_kwargs = dict(
    actor_label=f"actor_{tasks_file}",
    evaluator_label=f"evaluator_{tasks_file}",
    learner_label=f"learner_{FLAGS.agent}",
  )

  experiment = experiment_builder.build_online_experiment_config(
    experiment_config_inputs=experiment_config_inputs,
    agent=FLAGS.agent,
    log_dir=log_dir,
    wandb_init_kwargs=wandb_init_kwargs,
    logger_factory_kwargs=logger_factory_kwargs,
    debug=debug,
    **kwargs,
  )
  if FLAGS.run_distributed:
    program = experiments.make_distributed_experiment(
        experiment=experiment,
        num_actors=FLAGS.num_actors)

    local_resources = {
        "actor": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "evaluator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "counter": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "replay": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
        "coordinator": PythonProcess(env={"CUDA_VISIBLE_DEVICES": ""}),
    }
    lp.launch(program,
              lp.LaunchType.LOCAL_MULTI_PROCESSING,
              terminal='current_terminal',
              local_resources=local_resources)
  else:
    experiments.run_experiment(experiment=experiment)

def sweep(search: str = 'default'):
  if search == 'default':
    space = [
        {
            "seed": tune.grid_search([1]),
            "agent": tune.grid_search(['uvfa_flat']),
        }
    ]
  else:
    raise NotImplementedError(search)

  return space


def setup_wandb_init_kwargs():
  search = FLAGS.search or 'default'
  wandb_init_kwargs = dict(
      project=FLAGS.wandb_project,
      entity=FLAGS.wandb_entity,
      notes=FLAGS.wandb_notes,
      save_code=False,
  )
  if FLAGS.train_single:
    # overall group
    wandb_init_kwargs['group'] = FLAGS.wandb_group if FLAGS.wandb_group else f"{search}_{FLAGS.search}"
  else:
    if FLAGS.wandb_group:
      logging.info(f'IGNORING `wandb_group`. This will be set using the current `search`')
    wandb_init_kwargs['group'] = search

  if FLAGS.wandb_name:
    wandb_init_kwargs['name'] = FLAGS.wandb_name

  if not FLAGS.use_wandb:
    wandb_init_kwargs = None

def main(_):
  # -----------------------
  # env setup
  # -----------------------
  default_env_kwargs = dict(
      tasks_file=FLAGS.tasks_file,
      room_size=FLAGS.room_size,
      num_dists=1,
      partial_obs=False,
  )
  agent_config_kwargs = dict()
  if FLAGS.debug:
    agent_config_kwargs.update(dict(
      show_gradients=1,
      samples_per_insert=1,
      min_replay_size=100,
    ))
    default_env_kwargs.update(dict(
    ))

  # -----------------------
  # wandb setup
  # -----------------------
  wandb_init_kwargs = setup_wandb_init_kwargs()
  if FLAGS.train_single:
    train_single(
      wandb_init_kwargs=wandb_init_kwargs,
      default_env_kwargs=default_env_kwargs,
      agent_config_kwargs=agent_config_kwargs)
  else:
    parallel.run(
      name='babyai_online_trainer',
      wandb_init_kwargs=wandb_init_kwargs,
      default_env_kwargs=default_env_kwargs,
      use_wandb=FLAGS.use_wandb,
      debug=FLAGS.debug,
      space=sweep(FLAGS.search, FLAGS.agent),
      make_program_command=functools.partial(
        parallel.make_program_command,
        filename='trainer.py',
        run_distributed=FLAGS.run_distributed,
        num_actors=FLAGS.num_actors),
    )

if __name__ == '__main__':
  app.run(main)