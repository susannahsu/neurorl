"""
Running experiments:
--------------------

# DEBUGGING, single stream
python -m ipdb -c continue configs/minigrid_trainer.py \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --use_wandb=False \
  --wandb_entity=wcarvalho92 \
  --wandb_project=minigrid_debug \
  --search='baselines'

# DEBUGGING, single stream, disable just-in-time compilation
JAX_DISABLE_JIT=1 python -m ipdb -c continue configs/minigrid_trainer.py \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --use_wandb=False \
  --wandb_entity=wcarvalho92 \
  --wandb_project=minigrid_debug \
  --search='baselines'


# DEBUGGING, launching jobs on slurm: see `sweep` fn
python -m ipdb -c continue configs/minigrid_trainer.py \
  --parallel='sbatch' \
  --debug_parallel=True \
  --run_distributed=False \
  --use_wandb=True \
  --wandb_entity=wcarvalho92 \
  --wandb_project=minigrid_debug \
  --search='baselines'


# launching jobs on slurm: see `sweep` fn
python configs/minigrid_trainer.py \
  --parallel='sbatch' \
  --run_distributed=True \
  --use_wandb=True \
  --partition=kempner \
  --account=kempner_fellows \
  --wandb_entity=wcarvalho92 \
  --wandb_project=minigrid \
  --search='muzero'

"""
import functools 

import dataclasses
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

from td_agents import basics

from library.dm_env_wrappers import GymWrapper
import library.env_wrappers as env_wrappers
import library.experiment_builder as experiment_builder
import library.parallel as parallel
import library.utils as utils

flags.DEFINE_string('config_file', '', 'config file')
flags.DEFINE_string('search', 'default', 'which search to use.')
flags.DEFINE_string(
    'parallel', 'none', "none: run 1 experiment. sbatch: run many experiments with SBATCH. ray: run many experiments with say. use sbatch with SLUM or ray otherwise.")
flags.DEFINE_bool(
    'debug', False, 'If in debugging mode, only 1st config is run.')
flags.DEFINE_bool(
    'make_path', True, 'Create a path under `FLAGS>folder` for the experiment')

FLAGS = flags.FLAGS

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

def setup_experiment_inputs(
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

  if agent == 'qlearning':
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
            config=config)

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
                  step_size=config.scalar_step_size,
                  max_value=config.max_scalar_value,
                  tx_pair=config.tx_pair,
              )
    config.num_bins = discretizer.num_bins

    builder = basics.Builder(
      config=config,
      get_actor_core_fn=functools.partial(
          muzero.get_actor_core,
          mcts_policy=mcts_policy,
          discretizer=discretizer,
      ),
      optimizer_cnstr=muzero.muzero_optimizer_constr,
      LossFn=muzero.MuZeroLossFn(
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
          model_reward_coef=config.model_reward_coef,
      ))
    network_factory = functools.partial(muzero.make_minigrid_networks, config=config)
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
      samples_per_insert=4.0,
      min_replay_size=1_000,
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
    assert agent_config_kwargs['samples_per_insert'] > 0

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
  if search == 'baselines':
    space = [
        {
            "group": tune.grid_search(['run-3-babyai-torso']),
            "agent": tune.grid_search(['muzero']),
            "seed": tune.grid_search([1]),
            "env.level": tune.grid_search([
                "BabyAI-GoToRedBallNoDists-v0",
                "BabyAI-GoToObjS6-v1",
            ]),
        }
    ]
  elif search == 'muzero':
    space = [
        # {
        #     "group": tune.grid_search(['muzero-run-2']),
        #     "agent": tune.grid_search(['muzero']),
        #     "seed": tune.grid_search([1]),
        #     "warmup_steps": tune.grid_search([1_000]),
        #     "lr_transition_steps": tune.grid_search([1_000, 100_000]),
        #     "env.level": tune.grid_search([
        #         "BabyAI-GoToRedBallNoDists-v0",
        #     ]),
        # },
        {
            "group": tune.grid_search(['muzero-run-6']),
            "agent": tune.grid_search(['muzero']),
            "seed": tune.grid_search([1]),
            # "samples_per_insert_tolerance_rate": tune.grid_search([.5, 1., 10.]),
            "samples_per_insert": tune.grid_search([50., 25.]),
            "env.level": tune.grid_search([
                "BabyAI-GoToRedBallNoDists-v0",
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
