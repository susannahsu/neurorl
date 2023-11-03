
import functools 

from absl import flags
from absl import app
from absl import logging
import os
from ray import tune
from launchpad.nodes.python.local_multi_processing import PythonProcess
import launchpad as lp

from acme import wrappers as acme_wrappers
from acme.jax import experiments
import gymnasium
import dm_env
import minigrid

from dm_env_wrappers import GymWrapper
import envs
import env_wrappers
import experiment_builder
import experiment_logger
import parallel
import utils


flags.DEFINE_string('search', 'default', 'which search to use.')
flags.DEFINE_bool(
    'parallel', False, 'Run many or 1 experiments')
flags.DEFINE_bool(
    'make_path', False, 'Create a path under `FLAGS>folder` for the experiment')
flags.DEFINE_bool(
    'auto_name_wandb', False, 'automatically name wandb.')
FLAGS = flags.FLAGS

def make_environment(seed: int,
                     object_options: bool = True,
                     evaluation: bool = False) -> dm_env.Environment:
  """Loads environments.
  
  Args:
      evaluation (bool, optional): whether evaluation.
  
  Returns:
      dm_env.Environment: Multitask environment is returned.
  """
  del seed


  fixed_door_locs = False if evaluation else True

  # create gymnasium.Gym environment
  env = envs.KeyRoom(
    num_dists=0,
    fixed_door_locs=fixed_door_locs)
  
  # add wrappers
  gym_wrappers = [env_wrappers.DictObservationSpaceWrapper]
  if object_options:
    gym_wrappers.append(functools.partial(
      env_wrappers.GotoOptionsWrapper, use_options=object_options))
  
  # MUST GO LAST. GotoOptionsWrapper exploits symbolic obs
  gym_wrappers.append(functools.partial(
    minigrid.wrappers.RGBImgPartialObsWrapper, tile_size=8))

  for wrapper in gym_wrappers:
    env = wrapper(env)

  # convert to dm_env.Environment enironment
  env = GymWrapper(env)

  # add acme wrappers
  wrapper_list = [
    acme_wrappers.ObservationActionRewardWrapper,
    acme_wrappers.SinglePrecisionWrapper,
  ]

  return acme_wrappers.wrap_all(env, wrapper_list)

def setup_experiment_inputs(
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
  agent = agent_config_kwargs.get('agent', 'flat_usfa')
  if agent == 'flat_usfa':
    from td_agents import basics
    from td_agents import sf_agents
    config = sf_agents.Config(**config_kwargs)
    builder = basics.Builder(
      config=config,
      get_actor_core_fn=sf_agents.get_actor_core,
      LossFn=sf_agents.UsfaLossFn(
        discount=config.discount,
        importance_sampling_exponent=config.importance_sampling_exponent,
        burn_in_length=config.burn_in_length,
        max_replay_size=config.max_replay_size,
        max_priority_weight=config.max_priority_weight,
        bootstrap_n=config.bootstrap_n,
      ))
    # NOTE: main differences below
    network_factory = functools.partial(
            sf_agents.make_minigrid_networks, config=config)
    env_kwargs['object_options'] = False  # has no mechanism to select from object options since dependent on what agent sees
  elif agent == 'object_usfa':
    from td_agents import basics
    from td_agents import sf_agents
    config = sf_agents.Config(**config_kwargs)
    builder = basics.Builder(
      config=config,
      get_actor_core_fn=sf_agents.get_actor_core,
      LossFn=sf_agents.UsfaLossFn(
        discount=config.discount,
        importance_sampling_exponent=config.importance_sampling_exponent,
        burn_in_length=config.burn_in_length,
        max_replay_size=config.max_replay_size,
        max_priority_weight=config.max_priority_weight,
        bootstrap_n=config.bootstrap_n,
      ))
    # NOTE: main differences below
    network_factory = functools.partial(
            sf_agents.make_object_oriented_minigrid_networks, config=config)
    env_kwargs['object_options'] = True  # has no mechanism to select from object options since dependent on what agent sees
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
        reset=50 if not debug else 5),
      ]

  return experiment_builder.OnlineExperimentConfigInputs(
    agent=agent,
    agent_config=config,
    final_env_kwargs=env_kwargs,
    builder=builder,
    network_factory=network_factory,
    environment_factory=environment_factory,
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
    agent_config_kwargs=agent_config_kwargs,
    agent_config_file=FLAGS.agent_config,
    env_kwargs=default_env_kwargs,
    env_config_file=FLAGS.env_config,
    debug=debug)

  log_dir = FLAGS.folder or os.environ.get('RL_RESULTS_DIR', '/tmp/rl_results_dir')
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

  logger_factory_kwargs = dict(
    actor_label="actor",
    evaluator_label="evaluator",
    learner_label="learner",
  )

  experiment = experiment_builder.build_online_experiment_config(
    experiment_config_inputs=experiment_config_inputs,
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
    controller = lp.launch(program,
              lp.LaunchType.LOCAL_MULTI_PROCESSING,
              terminal='current_terminal',
              local_resources=local_resources)
    controller.wait(return_on_first_completed=True)
    controller._kill()

  else:
    experiments.run_experiment(experiment=experiment)

def extract_first_config(grid_search_space):
  """Extract the very first possible setting from the search space."""
  first_config = {}
  if isinstance(grid_search_space, list):
    grid_search_space = grid_search_space[0]
  for param_name, param_values in grid_search_space.items():
      first_value = next(iter(param_values.values()))[0]
      first_config[param_name] = first_value
  return first_config

def setup_wandb_init_kwargs():
  if not FLAGS.use_wandb:
    return None

  wandb_init_kwargs = dict(
      project=FLAGS.wandb_project,
      entity=FLAGS.wandb_entity,
      notes=FLAGS.wandb_notes,
      save_code=False,
  )
  search = FLAGS.search or 'default'

  if FLAGS.parallel:
    wandb_init_kwargs['group'] = search
  else:
    wandb_init_kwargs['group'] = FLAGS.wandb_group or search

  if FLAGS.wandb_name:
    wandb_init_kwargs['name'] = FLAGS.wandb_name

  return wandb_init_kwargs

def sweep(search: str = 'default'):
  if search == 'default':
    space = [
        {
            "seed": tune.grid_search([1]),
            "agent": tune.grid_search(['flat_usfa']),
        }
    ]
  else:
    raise NotImplementedError(search)

  return space

def main(_):
  default_env_kwargs = dict()
  agent_config_kwargs = dict()
  if FLAGS.debug or not FLAGS.parallel:
    agent_config_kwargs.update(dict(
      samples_per_insert=1,
      min_replay_size=100,
    ))
    default_env_kwargs.update(dict(
    ))

  # -----------------------
  # wandb setup
  # -----------------------
  wandb_init_kwargs = setup_wandb_init_kwargs()
  if FLAGS.parallel:
    parallel.run(
      wandb_init_kwargs=wandb_init_kwargs,
      default_env_kwargs=default_env_kwargs,
      use_wandb=FLAGS.use_wandb,
      debug=FLAGS.debug,
      space=sweep(FLAGS.search),
      make_program_command=functools.partial(
        parallel.make_program_command,
        filename='trainer.py',
        run_distributed=FLAGS.run_distributed,
        num_actors=FLAGS.num_actors),
    )
  else:
    first_config = extract_first_config(sweep(FLAGS.search))
    agent_config_kwargs.update(**first_config)
    train_single(
      wandb_init_kwargs=wandb_init_kwargs,
      default_env_kwargs=default_env_kwargs,
      agent_config_kwargs=agent_config_kwargs)

if __name__ == '__main__':
  app.run(main)
