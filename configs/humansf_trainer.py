"""
Running experiments:
- multiple distributed experiments in parallel:
    python trainer.py --search='default'

  - single asynchronous experiment with actor/learner/evaluator:
    python trainer.py --search='default' --run_distributed=True --debug=True

  - single synchronous experiment (most useful for debugging):
    python trainer.py --search='default' --run_distributed=False --debug=True

Change "search" to what you want to search over.

"""
import functools 

from typing import Callable, Optional

from enum import Enum

from absl import flags
from absl import app
from absl import logging
import os
from ray import tune
from launchpad.nodes.python.local_multi_processing import PythonProcess
import launchpad as lp

from acme import wrappers as acme_wrappers
from acme.jax import experiments
import dm_env

from lib.dm_env_wrappers import GymWrapper
import lib.env_wrappers as env_wrappers
import lib.experiment_builder as experiment_builder
import lib.experiment_logger as experiment_logger
import lib.parallel as parallel
import lib.utils as utils

import envs.key_room as key_room
from envs.key_room_objects_test import (
  ObjectTestTask,
  KeyRoomObjectTest,
  ObjectCountObserver,
)


flags.DEFINE_string('config_file', '', 'config file')
flags.DEFINE_string('search', 'default', 'which search to use.')
flags.DEFINE_string(
    'parallel', 'none', "none: run 1 experiment. sbatch: run many experiments with SBATCH. ray: run many experiments with say. use sbatch with SLUM or ray otherwise.")
flags.DEFINE_bool(
    'debug', False, 'If in debugging mode, only 1st config is run.')
flags.DEFINE_bool(
    'make_path', True, 'Create a path under `FLAGS>folder` for the experiment')
flags.DEFINE_bool(
    'auto_name_wandb', True, 'automatically name wandb.')
FLAGS = flags.FLAGS


class TestOptions(Enum):
  shape = 0
  color = 1
  ambigious = 2


def make_keyroom_object_test_env(seed: int,
                     setting: TestOptions,
                     room_size: int = 6,
                     evaluation: bool = False,
                     object_options: bool = True,
                     **kwargs) -> dm_env.Environment:
  """Loads environments.
  
  Args:
      evaluation (bool, optional): whether evaluation.
  
  Returns:
      dm_env.Environment: Multitask environment is returned.
  """
  del seed

  if setting == TestOptions.shape.value:
    # in this setting, the initial shape indicates the task color
    train_tasks = []
    test_tasks = []
    for c in ['blue', 'yellow']:
        train_tasks.append(
          ObjectTestTask(
            source='shape', init='ball', floor=c, w='box'))
        test_tasks.append(
          ObjectTestTask(
            source='shape', init='ball', floor=c, w='ball'))

        train_tasks.append(
          ObjectTestTask(
            source='shape', init='box', floor=c, w='ball'))
        test_tasks.append(
          ObjectTestTask(
            source='shape', init='box', floor=c, w='box'))

  elif setting == TestOptions.color.value:
    # in this setting, the floor color indicates the task color
    train_tasks = [
      ObjectTestTask(floor='blue', init='ball', w='box'),
      ObjectTestTask(floor='blue', init='box', w='box'),
      ObjectTestTask(floor='yellow', init='ball', w='ball'),
      ObjectTestTask(floor='yellow', init='box', w='ball'),
    ]

    test_tasks = [
      ObjectTestTask(floor='blue', init='ball', w='ball'),
      ObjectTestTask(floor='blue', init='box', w='ball'),
      ObjectTestTask(floor='yellow', init='ball', w='box'),
      ObjectTestTask(floor='yellow', init='box', w='box'),
    ]

  elif setting == TestOptions.ambigious.value:
    # in this setting, it's not clear whether floor color or initial shape indicates the task color
    floor2task_color = {
        'red': 'blue',
        'green': 'yellow',
    }

    train_tasks = [
        ObjectTestTask(floor='red', init='ball', w='box', floor2task_color=floor2task_color),
        ObjectTestTask(floor='green', init='box', w='ball', floor2task_color=floor2task_color),
    ]

    test_tasks = [
        ObjectTestTask(floor='red', init='ball', w='ball', floor2task_color=floor2task_color),
        ObjectTestTask(floor='red', init='box', w='ball', floor2task_color=floor2task_color),
        ObjectTestTask(floor='green', init='box', w='box', floor2task_color=floor2task_color),
        ObjectTestTask(floor='green', init='ball', w='box', floor2task_color=floor2task_color),
    ]

  else:
    raise NotImplementedError(setting)

  room_colors = list(set([t.goal_color() for t in train_tasks]))

  # create gymnasium.Gym environment
  env = KeyRoomObjectTest(
    room_size=room_size,
    tasks=test_tasks if evaluation else train_tasks,
    room_colors=room_colors,
    **kwargs)

  ####################################
  # Gym wrappers
  ####################################
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

  ####################################
  # ACME wrappers
  ####################################
  # add acme wrappers
  wrapper_list = [
    acme_wrappers.ObservationActionRewardWrapper,
    acme_wrappers.SinglePrecisionWrapper,
  ]

  return acme_wrappers.wrap_all(env, wrapper_list)

def setup_experiment_inputs(
    make_environment_fn: Callable,
    env_get_task_name: Optional[Callable[[dm_env.Environment], str]] = None,
    agent_config_kwargs: dict=None,
    env_kwargs: dict=None,
    debug: bool = False,
  ):
  """Setup."""
  config_kwargs = agent_config_kwargs or dict()
  env_kwargs = env_kwargs or dict()

  # -----------------------
  # load agent config, builder, network factory
  # -----------------------
  agent = agent_config_kwargs.get('agent', '')
  assert agent != '', 'please set agent'
  
  if agent == 'flat_q':
    from td_agents import basics
    from td_agents import q_learning
    import haiku as hk
    config = q_learning.Config(**config_kwargs)
    builder = basics.Builder(
      config=config,
      get_actor_core_fn=functools.partial(
        basics.get_actor_core,
        linear_epsilon=config.linear_epsilon,
      ),
      LossFn=q_learning.R2D2LossFn(
        discount=config.discount,
        
        importance_sampling_exponent=config.importance_sampling_exponent,
        burn_in_length=config.burn_in_length,
        max_replay_size=config.max_replay_size,
        max_priority_weight=config.max_priority_weight,
        bootstrap_n=config.bootstrap_n,
      ))
    # NOTE: main differences below
    network_factory = functools.partial(
            q_learning.make_minigrid_networks,
            config=config,
            task_encoder=lambda obs: hk.Linear(128)(obs['task']))

    env_kwargs['object_options'] = False  # has no mechanism to select from object options since dependent on what agent sees

  elif agent == 'flat_usfa':
    from td_agents import basics
    from td_agents import usfa
    config = usfa.Config(**config_kwargs)
    builder = basics.Builder(
      config=config,
      get_actor_core_fn=functools.partial(
        basics.get_actor_core,
        extract_q_values=lambda preds: preds.q_values,
        linear_epsilon=config.linear_epsilon,
        ),
      LossFn=usfa.UsfaLossFn(
        discount=config.discount,

        importance_sampling_exponent=config.importance_sampling_exponent,
        burn_in_length=config.burn_in_length,
        max_replay_size=config.max_replay_size,
        max_priority_weight=config.max_priority_weight,
        bootstrap_n=config.bootstrap_n,
      ))
    # NOTE: main differences below
    network_factory = functools.partial(
            usfa.make_minigrid_networks, config=config)
    env_kwargs['object_options'] = False  # has no mechanism to select from object options since dependent on what agent sees
  elif agent == 'object_usfa':
    from td_agents import basics
    from td_agents import usfa
    config = usfa.Config(**config_kwargs)
    builder = basics.Builder(
      config=config,
      get_actor_core_fn=functools.partial(
        basics.get_actor_core,
        extract_q_values=lambda preds: preds.q_values,
        linear_epsilon=config.linear_epsilon,
        ),
      LossFn=usfa.UsfaLossFn(
        discount=config.discount,

        importance_sampling_exponent=config.importance_sampling_exponent,
        burn_in_length=config.burn_in_length,
        max_replay_size=config.max_replay_size,
        max_priority_weight=config.max_priority_weight,
        bootstrap_n=config.bootstrap_n,
      ))
    # NOTE: main differences below
    network_factory = functools.partial(
            usfa.make_object_oriented_minigrid_networks, config=config)
    env_kwargs['object_options'] = True  # has no mechanism to select from object options since dependent on what agent sees
  else:
    raise NotImplementedError(agent)

  # -----------------------
  # load environment factory
  # -----------------------

  environment_factory = functools.partial(
    make_environment_fn,
    **env_kwargs)

  # -----------------------
  # setup observer factory for environment
  # -----------------------
  test_setting = TestOptions(env_kwargs['setting']).name
  observers = [
    # this logs the average every reset=50 episodes (instead of every episode)
    utils.LevelAvgReturnObserver(
      reset=50 if not debug else 5,
      get_task_name=env_get_task_name,
      ),
    ObjectCountObserver(
      reset=1000 if not debug else 5,
      prefix=f'Images-{test_setting}',
      agent_name=agent,
      get_task_name=env_get_task_name),
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
    env_kwargs: dict = None,
    wandb_init_kwargs: dict = None,
    agent_config_kwargs: dict = None,
    log_dir: str = None,
    num_actors: int = 1,
    run_distributed: bool = False,
):

  debug = FLAGS.debug

  experiment_config_inputs = setup_experiment_inputs(
    make_environment_fn=make_keyroom_object_test_env,
    env_get_task_name= lambda env: env.unwrapped.task.goal_name(),
    agent_config_kwargs=agent_config_kwargs,
    env_kwargs=env_kwargs,
    debug=debug)

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
    debug=debug
  )
  if run_distributed:
    program = experiments.make_distributed_experiment(
        experiment=experiment,
        num_actors=num_actors)

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

def setup_wandb_init_kwargs():
  if not FLAGS.use_wandb:
    return dict()

  wandb_init_kwargs = dict(
      project=FLAGS.wandb_project,
      entity=FLAGS.wandb_entity,
      notes=FLAGS.wandb_notes,
      name=FLAGS.wandb_name,
      group=FLAGS.search or 'default',
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
      samples_per_insert=1,
      min_replay_size=100,
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

  if not run_distributed:
    agent_config_kwargs['samples_per_insert'] = 1

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
  if search == 'flat':
    space = [
        {
            "agent": tune.grid_search(['flat_usfa']),
            "seed": tune.grid_search([1]),
            "group": tune.grid_search(['sf-test-4']),
            "env.setting": tune.grid_search([0]),
            # "env.transfer_task_option": tune.grid_search([0]),
            "linear_epsilon": tune.grid_search([True, False]),
        },
    ]
  elif search == 'speed':
    space = [
        {
            "agent": tune.grid_search(['flat_q', 'flat_usfa']),
            "seed": tune.grid_search([1]),
            "group": tune.grid_search(['speed-test-7']),
            "samples_per_insert": tune.grid_search([10]),
            "env.setting": tune.grid_search([0]),
        },
    ]
  elif search == 'objects':
    space = [
        {
            "seed": tune.grid_search([5,6,7,8]),
            "agent": tune.grid_search(['object_usfa']),
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
