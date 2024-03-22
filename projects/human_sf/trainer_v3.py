"""
Running experiments:

# DEBUGGING, single stream
python -m ipdb -c continue projects/human_sf/trainer_v3.py \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --use_wandb=False \
  --wandb_entity=wcarvalho92 \
  --wandb_project=human_sf_v2_debug \
  --search='usfa_dyna2'

# DEBUGGING, without jit
JAX_DISABLE_JIT=1 python -m ipdb -c continue projects/human_sf/trainer_v3.py \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --use_wandb=False \
  --wandb_entity=wcarvalho92 \
  --wandb_project=human_sf_v2_debug \
  --search='usfa'


# DEBUGGING, parallel
python -m ipdb -c continue projects/human_sf/trainer_v3.py \
  --parallel='sbatch' \
  --debug_parallel=True \
  --run_distributed=False \
  --use_wandb=True \
  --wandb_entity=wcarvalho92 \
  --wandb_project=human_sf_v2_debug \
  --search='usfa'


# running, parallel
python projects/human_sf/trainer_v3.py \
  --parallel='sbatch' \
  --run_distributed=True \
  --use_wandb=True \
  --partition=kempner \
  --account=kempner_fellows \
  --wandb_entity=wcarvalho92 \
  --wandb_project=human_sf_v2 \
  --max_concurrent=12 \
  --time='0-02:00:00' \
  --search='usfa_dyna2'

Change "search" to what you want to search over.

"""
import functools 
import json
from typing import Callable, Optional, Tuple

from enum import Enum

from absl import flags
from absl import app
from absl import logging
import dataclasses
import os
from ray import tune
from launchpad.nodes.python.local_multi_processing import PythonProcess
import launchpad as lp

from acme.agents.jax import actor_core as actor_core_lib
from acme import wrappers as acme_wrappers
from acme.jax import experiments
import dm_env
import haiku as hk
import jax

import minigrid

from projects.human_sf.minigrid_goto_wrapper import GotoOptionsWrapper
from envs.minigrid_wrappers import DictObservationSpaceWrapper
from library.dm_env_wrappers import GymWrapper
import library.experiment_builder as experiment_builder
import library.parallel as parallel
import library.utils as utils

from td_agents import basics

from projects.human_sf import q_learning
from projects.human_sf import object_q_learning
from projects.human_sf import muzero
from projects.human_sf import object_muzero
from projects.human_sf import scalar_muzero
from projects.human_sf import cat_usfa
from projects.human_sf import usfa_offtask as usfa
from projects.human_sf import usfa_dyna_offtask as usfa_dyna
from projects.human_sf import object_usfa_offtask as object_usfa
from projects.human_sf import object_usfa_dyna_offtask as object_usfa_dyna
from projects.human_sf import utils as human_proj_utils
from projects.human_sf.minigrid_room_view_wrapper import RGBImgRoomObsWrapper, OneHotRoomObsWrapper

from projects.human_sf import key_room_v3 as key_room


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

@dataclasses.dataclass
class QlearningConfig(q_learning.Config):
  samples_per_insert: float = 10.0
  importance_sampling_exponent: float = 0.0

@dataclasses.dataclass
class ObjectQlearningConfig(object_q_learning.Config):
  samples_per_insert: float = 10.0
  importance_sampling_exponent: float = 0.0


@dataclasses.dataclass
class UsfaConfig(usfa.Config):

  # arch
  # state_dim: int = 256
  # final_conv_dim: int = 16
  # conv_flat_dim: Optional[int] = 0
  # sf_layers : Tuple[int]=(1024, 1024)
  # policy_layers : Tuple[int]=(128, 128)

  # learner
  importance_sampling_exponent: float = 0.0
  samples_per_insert: float = 10.0
  # sf_coeff: float = 1e3
  # q_coeff: float = 1.0

  # combine_policy: str = 'sum'
  # head: str = 'monolithic'

  # # eval actor
  # eval_task_support: str = "train"  # options:

@dataclasses.dataclass
class MuZeroConfig(muzero.Config):
  """Configuration options for MuZero agent."""
  trace_length: int = 40
  min_scalar_value: Optional[float] = None
  num_bins: Optional[int] = 81  # number of bins for two-hot rep
  scalar_step_size: Optional[float] = None  # step size between bins

def make_keyroom_env(
    seed: int,
    room_size: int = 7,
    evaluation: bool = False,
    evaluate_key_only: bool = True,
    object_options: bool = False,
    flat_task: bool = True,
    steps_per_room: int = 100,
    swap_episodes: int = 0,
    maze_idx: int = 0,
    num_task_rooms: int = 2,
    color_rooms: bool = False,
    key_room_rewards = (.25, .5),
    basic_only: int = 0,
    auto_pickup: bool = True,
    tile_size: int = 8,
    debug: bool = False,
    symbolic: bool = False,
    **kwargs) -> dm_env.Environment:
  """Loads environments.
  
  Args:
      evaluation (bool, optional): whether evaluation.
  
  Returns:
      dm_env.Environment: Multitask environment is returned.
  """
  del seed
  if debug:
    pass
    # steps_per_room = 30
    # basic_only = 0
    # num_task_rooms = 4

  json_file = 'projects/human_sf/maze_pairs.json'
  with open(json_file, 'r') as file:
      maze_configs = json.load(file)
  maze_config = maze_configs[maze_idx]

  # create gymnasium.Gym environment
  env = key_room.KeyRoom(
    room_size=room_size,
    maze_config=maze_config,
    max_steps_per_room=steps_per_room,
    flat_task=flat_task,
    num_task_rooms=num_task_rooms,
    swap_episodes=swap_episodes,
    color_rooms=color_rooms,
    basic_only=basic_only,
    key_room_rewards=key_room_rewards,
    training= not evaluation,
    evaluate_key_only=evaluate_key_only,
    **kwargs)

  ####################################
  # Gym wrappers
  ####################################
  gym_wrappers = [DictObservationSpaceWrapper]

  if symbolic:
    gym_wrappers.append(functools.partial(
      OneHotRoomObsWrapper,
      tile_size=tile_size,
      include_images=evaluation,
      ))
  else:
    gym_wrappers.append(functools.partial(
      RGBImgRoomObsWrapper, tile_size=tile_size))

  if object_options:
    gym_wrappers.append(functools.partial(
      GotoOptionsWrapper,
      max_options=15,
      use_options=True,
      auto_pickup=auto_pickup,
      ))

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

def make_q_loss_fn(config):
  return q_learning.R2D2LossFn(
          discount=config.discount,
          importance_sampling_exponent=config.importance_sampling_exponent,
          burn_in_length=config.burn_in_length,
          max_replay_size=config.max_replay_size,
          max_priority_weight=config.max_priority_weight,
          bootstrap_n=config.bootstrap_n,
        )

def make_cat_sf_loss_fn(config, **kwargs):
  discretizer = utils.Discretizer(
    num_bins=config.num_bins,
    min_value=config.min_scalar_value,
    max_value=config.max_scalar_value,
    tx_pair=config.tx_pair)
  return cat_usfa.UsfaLossFn(
    discount=config.discount,
    importance_sampling_exponent=config.importance_sampling_exponent,
    burn_in_length=config.burn_in_length,
    max_replay_size=config.max_replay_size,
    max_priority_weight=config.max_priority_weight,
    bootstrap_n=config.bootstrap_n,
    sf_coeff=config.sf_coeff,
    loss_fn=config.loss_fn,
    discretizer=discretizer,
    lambda_=config.sf_lambda,
    sum_cumulants=config.sum_cumulants,
    # weight_type=config.weight_type,
    # combination=config.combination,
    # off_task_weight=config.off_task_weight,
    q_coeff=config.q_coeff,
    **kwargs
  )

def make_sf_loss_fn(config, **kwargs):
  return usfa.MultitaskUsfaLossFn(
    discount=config.discount,
    importance_sampling_exponent=config.importance_sampling_exponent,
    burn_in_length=config.burn_in_length,
    max_replay_size=config.max_replay_size,
    max_priority_weight=config.max_priority_weight,
    bootstrap_n=config.bootstrap_n,
    sf_coeff=config.sf_coeff,
    loss_fn=config.loss_fn,
    lambda_=config.sf_lambda,
    sum_cumulants=config.sum_cumulants,
    weight_type=config.weight_type,
    combination=config.combination,
    off_task_weight=config.off_task_weight,
    q_coeff=config.q_coeff,
    **kwargs
  )

def make_sf_dyna_loss_fn(config, **kwargs):
  return usfa_dyna.MtrlDynaUsfaLossFn(
    discount=config.discount,
    importance_sampling_exponent=config.importance_sampling_exponent,
    burn_in_length=config.burn_in_length,
    max_replay_size=config.max_replay_size,
    max_priority_weight=config.max_priority_weight,
    bootstrap_n=config.bootstrap_n,
    task_coeff=config.task_coeff,
    dyna_coeff=config.dyna_coeff,
    model_coeff=config.model_coeff,
    model_sf_coeff=config.model_sf_coeff,
    task_weighted_dyna=config.task_weighted_dyna,
    policy_rep=config.policy_rep,
    # gpi_eval_model=config.gpi_eval_model,
    model_unweighted_coeff=config.model_unweighted_coeff,
    model_discount=config.model_discount,
    stop_dyna_state_grad=config.stop_dyna_state_grad,
    # task_weighted_cumulant_loss=config.task_weighted_cumulant_loss,
    mask_zero_features=config.mask_zero_features,
    backup_train_task=config.backup_train_task,
    feature_coeff=config.feature_coeff,
    loss_fn=config.loss_fn,
    n_actions_dyna=config.n_actions_dyna,
    n_tasks_dyna=config.n_tasks_dyna,
    binary_feature_loss=config.binary_feature_loss,
    task_weighted_model=config.task_weighted_model,
    weighted_coeff=config.weighted_coeff,
    unweighted_coeff=config.unweighted_coeff,
    simulation_steps=config.simulation_steps,
    cat_coeff=config.cat_coeff,
    **kwargs)

def setup_experiment_inputs(
    make_environment_fn: Callable,
    env_get_task_name: Optional[Callable[[dm_env.Environment], str]] = None,
    agent_config_kwargs: dict=None,
    env_kwargs: dict=None,
    debug: bool = False,
    log_dir: str = '.',
  ):
  """Setup."""
  config_kwargs = agent_config_kwargs or dict()
  env_kwargs = env_kwargs or dict()

  # -----------------------
  # load agent config, builder, network factory
  # -----------------------
  agent = agent_config_kwargs.get('agent', '')
  assert agent != '', 'please set agent'

  #################################
  # Observers
  #################################
  action_names = {
    0: 'left',
    1: 'right',
    2: 'forward',
    3: 'pickup',
    4: 'drop',
    5: 'toggle',
    6: 'done',
    7: 'blue ball',
    8: 'blue box',
    9: 'green box',
    10: 'purple box',
    11: 'grey door',
    12: 'red door',
    13: 'grey key',
    14: 'red key'}

  q_observer  = q_learning.Observer(period=1 if debug else 4000)
  sf_observer = human_proj_utils.SFObserver(
    period=1 if debug else 4000,
    action_names=action_names,
    log_dir=log_dir)

  #################################
  # flat agents
  #################################

  symbolic = env_kwargs.get('symbolic', False)
  config_kwargs['symbolic'] = symbolic

  task_encoder = lambda obs: hk.nets.MLP(
    (128, 128), activate_final=True)(obs['task'])

  if agent == 'flat_q':
    # has no mechanism to select from object options since dependent on what agent sees
    env_kwargs['object_options'] = False

    config = QlearningConfig(**config_kwargs)
    builder = basics.Builder(
      config=config,
      get_actor_core_fn=basics.get_actor_core,
      ActorCls=functools.partial(basics.BasicActor, observers=[q_observer]),
      LossFn=make_q_loss_fn(config))
    network_factory = functools.partial(
      q_learning.make_minigrid_networks,
      config=config,
      task_encoder=task_encoder)

  elif agent == 'flat_usfa':
    # has no mechanism to select from object options since dependent on what agent sees
    env_kwargs['object_options'] = False

    config = UsfaConfig(**config_kwargs)
    builder = basics.Builder(
      config=config,
      ActorCls=functools.partial(basics.BasicActor, observers=[sf_observer]),
      get_actor_core_fn=usfa.get_actor_core,
      LossFn=make_sf_loss_fn(config))
    network_factory = functools.partial(
            usfa.make_minigrid_networks, config=config)

  elif agent == 'flat_cat_usfa':
    # has no mechanism to select from object options since dependent on what agent sees
    env_kwargs['object_options'] = False

    config = cat_usfa.Config(**config_kwargs)
    builder = basics.Builder(
      config=config,
      ActorCls=functools.partial(
        basics.BasicActor, observers=[sf_observer]),
      get_actor_core_fn=cat_usfa.get_actor_core,
      LossFn=make_cat_sf_loss_fn(config))
    network_factory = functools.partial(
            cat_usfa.make_minigrid_networks, config=config)

  elif agent == 'flat_usfa_dyna':
    # has no mechanism to select from object options since dependent on what agent sees
    env_kwargs['object_options'] = False

    config = usfa_dyna.Config(**config_kwargs)
    builder = basics.Builder(
      config=config,
      ActorCls=functools.partial(basics.BasicActor, observers=[sf_observer]),
      get_actor_core_fn=usfa_dyna.get_actor_core,
      LossFn=make_sf_dyna_loss_fn(config))
    network_factory = functools.partial(
            usfa_dyna.make_minigrid_networks, config=config)

  elif agent == 'flat_muzero':
    # has no mechanism to select from object options since dependent on what agent sees
    env_kwargs['object_options'] = False

    config = muzero.Config(**config_kwargs)

    import mctx
    mcts_act_policy = functools.partial(
        mctx.gumbel_muzero_policy,
        max_depth=config.max_sim_depth_eval,
        num_simulations=config.num_simulations,
        gumbel_scale=config.gumbel_scale)

    # currently using same policy in learning and acting
    mcts_learn_policy = functools.partial(
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
            muzero.muzero_policy_act_mcts_eval,
            mcts_policy=mcts_act_policy,
            discretizer=discretizer,
            mcts_train=config.mcts_train,
            mcts_eval=config.mcts_eval,
        ),
        optimizer_cnstr=muzero.muzero_optimizer_constr,
        LossFn=muzero.MuZeroLossFn(
            discount=config.discount,
            importance_sampling_exponent=config.importance_sampling_exponent,
            burn_in_length=config.burn_in_length,
            max_replay_size=config.max_replay_size,
            max_priority_weight=config.max_priority_weight,
            bootstrap_n=config.bootstrap_n,
            value_target_source=config.value_target_source,
            discretizer=discretizer,
            mcts_policy=mcts_learn_policy,
            simulation_steps=config.simulation_steps,
            reanalyze_ratio=config.reanalyze_ratio,
            root_policy_coef=config.root_policy_coef,
            root_value_coef=config.root_value_coef,
            model_policy_coef=config.model_policy_coef,
            model_value_coef=config.model_value_coef,
            model_reward_coef=config.model_reward_coef,
        ))
    network_factory = functools.partial(
        muzero.make_minigrid_networks,
        config=config,
        task_encoder=task_encoder)
  #################################
  # object centric agents
  #################################
  elif agent == 'object_q':
    # has no mechanism to select from object options since dependent on what agent sees
    env_kwargs['object_options'] = True

    config = ObjectQlearningConfig(**config_kwargs)
    builder = basics.Builder(
      config=config,
      get_actor_core_fn=object_q_learning.get_actor_core,
      ActorCls=functools.partial(basics.BasicActor, observers=[q_observer]),
      LossFn=make_q_loss_fn(config))
    network_factory = functools.partial(
      object_q_learning.make_minigrid_networks,
      config=config,
      task_encoder=task_encoder)

  elif agent == 'object_usfa':
    env_kwargs['object_options'] = True  # has no mechanism to select from object options since dependent on what agent sees

    config = UsfaConfig(**config_kwargs)
    builder = basics.Builder(
      config=config,
      get_actor_core_fn=object_usfa.get_actor_core,
      ActorCls=functools.partial(basics.BasicActor, observers=[sf_observer]),
      LossFn=make_sf_loss_fn(config))
    network_factory = functools.partial(
      object_usfa.make_minigrid_networks, config=config)

  elif agent == 'object_usfa_dyna':
    env_kwargs['object_options'] = True  # has no mechanism to select from object options since dependent on what agent sees

    config = usfa_dyna.Config(**config_kwargs)
    builder = basics.Builder(
      config=config,
      get_actor_core_fn=usfa_dyna.get_actor_core,
      ActorCls=functools.partial(basics.BasicActor, observers=[sf_observer]),
      LossFn=make_sf_dyna_loss_fn(config, action_mask=True))
    network_factory = functools.partial(
      object_usfa_dyna.make_minigrid_networks, config=config)

  elif agent == 'object_usfa_att':
    env_kwargs['object_options'] = True  # has no mechanism to select from object options since dependent on what agent sees
    env_kwargs['flat_task'] = False  # has no mechanism to select from object options since dependent on what agent sees
    raise NotImplementedError('uncomment below and finis')
    config = UsfaConfig(**config_kwargs)
    builder = basics.Builder(
      config=config,
      get_actor_core_fn=object_usfa.get_actor_core,
      ActorCls=functools.partial(basics.BasicActor, observers=[sf_observer]),
      LossFn=make_sf_loss_fn(config))
    # NOTE: main differences below
    network_factory = functools.partial(
      object_usfa.make_object_oriented_minigrid_networks,
      config=config,
      task_encoder=task_encoder,
      attribute_factorization=True)

  elif agent == 'object_muzero':
    # has no mechanism to select from object options since dependent on what agent sees
    env_kwargs['object_options'] = True

    config = object_muzero.Config(**config_kwargs)

    import mctx
    mcts_act_policy = functools.partial(
        mctx.gumbel_muzero_policy,
        max_depth=config.max_sim_depth_eval,
        num_simulations=config.num_simulations,
        gumbel_scale=config.gumbel_scale)

    # currently using same policy in learning and acting
    mcts_learn_policy = functools.partial(
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
            muzero.muzero_policy_act_mcts_eval,
            mcts_policy=mcts_act_policy,
            discretizer=discretizer,
            mcts_train=config.mcts_train,
        ),
        optimizer_cnstr=muzero.muzero_optimizer_constr,
        LossFn=muzero.MuZeroLossFn(
            discount=config.discount,
            importance_sampling_exponent=config.importance_sampling_exponent,
            burn_in_length=config.burn_in_length,
            max_replay_size=config.max_replay_size,
            max_priority_weight=config.max_priority_weight,
            bootstrap_n=config.bootstrap_n,
            value_target_source=config.value_target_source,
            discretizer=discretizer,
            mcts_policy=mcts_learn_policy,
            simulation_steps=config.simulation_steps,
            reanalyze_ratio=config.reanalyze_ratio,
            root_policy_coef=config.root_policy_coef,
            root_value_coef=config.root_value_coef,
            model_policy_coef=config.model_policy_coef,
            model_value_coef=config.model_value_coef,
            model_reward_coef=config.model_reward_coef,
            object_options_mask=True,
        ))
    network_factory = functools.partial(
        object_muzero.make_minigrid_networks,
        config=config,
        task_encoder=task_encoder)

  else:
    raise NotImplementedError(agent)

  # -----------------------
  # load environment factory
  # -----------------------

  environment_factory = functools.partial(
    make_environment_fn,
    debug=debug,
    tile_size=config.tile_size,
    **env_kwargs)

  # -----------------------
  # setup observer factory for environment
  # -----------------------
  observers = [
    # this logs the average every reset=50 episodes (instead of every episode)
    utils.LevelAvgReturnObserver(
      reset=50 if not debug else 5,
      get_task_name=env_get_task_name,
      ),
    # human_proj_utils.LevelEpisodeObserver(
    #   reset=2000 if not debug else 1,
    #   prefix='EpisodeObserver',
    #   get_task_name=env_get_task_name,
    #   get_action_names=lambda e: e.action_names,
    # ),
    key_room.ObjectCountObserver(
      reset=5000 if not debug else 5,
      prefix=f'Images',
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
  def env_get_task_name(env):
    return env.unwrapped.task_name
    # name += "\n"
    # name +=str(env.unwrapped.task.task_array)
    # name += "\n"

  experiment_config_inputs = setup_experiment_inputs(
    make_environment_fn=make_keyroom_env,
    env_get_task_name=env_get_task_name,
    agent_config_kwargs=agent_config_kwargs,
    env_kwargs=env_kwargs,
    log_dir=log_dir,
    debug=debug)

  logger_factory_kwargs = dict(
    actor_label=f"actor",
    evaluator_label=f"evaluator",
    learner_label=f"learner",
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

  if FLAGS.debug:
    agent_config_kwargs.update(dict(
      samples_per_insert=1,
      min_replay_size=40,
      sf_layers=(128,),
      policy_layers=(),
      max_replay_size=1_000,
      batch_size=2,
      trace_length=5,
      state_dim=32,
    ))
    env_kwargs.update(dict(
    ))

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
  if search == 'flat_q':
    space = [
        {
            "num_steps": tune.grid_search([5e6]),
            "agent": tune.grid_search(['flat_q']),
            "seed": tune.grid_search([5]),
            "group": tune.grid_search(['flat_q-5']),
            "dot": tune.grid_search([False]),
            "env.basic_only": tune.grid_search([1]),
        },
    ]
  elif search == 'flat_usfa':
    space = [
        {
            "num_steps": tune.grid_search([5e6]),
            "agent": tune.grid_search(['flat_usfa']),
            "seed": tune.grid_search([6]),
            "group": tune.grid_search(['flat-usfa-6']),
            "env.basic_only": tune.grid_search([1]),
        },
    ]
  elif search == 'flat_muzero':
    space = [
        {
            "num_steps": tune.grid_search([5e6]),
            "agent": tune.grid_search(['flat_muzero']),
            "seed": tune.grid_search([5]),
            "env.basic_only": tune.grid_search([1]),
            "group": tune.grid_search(['flat_muzero-6']),
        },
    ]
  elif search == 'object_q':
    space = [
        {
            "num_steps": tune.grid_search([30e6]),
            "agent": tune.grid_search(['object_q']),
            "seed": tune.grid_search([5]),
            "group": tune.grid_search(['object_q-2']),
            # "dot": tune.grid_search([False, True]),
            # "object_conditioned": tune.grid_search([False, True]),
            "env.basic_only": tune.grid_search([1]),
            # "trace_length": tune.grid_search([10, 20]),
        },
    ]
  elif search == 'object_muzero':
    space = [
        {
            "num_steps": tune.grid_search([30e6]),
            "agent": tune.grid_search(['object_muzero']),
            "seed": tune.grid_search([5]),
            "group": tune.grid_search(['object_muzero-2']),
            "env.basic_only": tune.grid_search([0]),
            # "trace_length": tune.grid_search([10, 20]),
        },
    ]
  elif search == 'usfa':
    space = [
        {
            # "num_steps": tune.grid_search([5e6]),
            # "env.basic_only": tune.grid_search([1]),
            "num_steps": tune.grid_search([20e6]),
            "env.num_task_rooms": tune.grid_search([1]),
            "agent": tune.grid_search([
              'object_usfa',
              'flat_usfa'
            ]),
            "seed": tune.grid_search([5]),
            "group": tune.grid_search(['usfa-35-off_task-object']),
            # "sf_layers": tune.grid_search([[128, 128], [256, 256], [512, 512]]),
            "off_task_weight": tune.grid_search([.01, .05]),
            # "head": tune.grid_search(['monolithic', 'independent']),
            'learning_support': tune.grid_search([
              'train_tasks',
              # 'eval', 
              ]),
            # 'sf_coeff': tune.grid_search([1.0, 0.0]),
        },
    ]
  elif search == 'cat_usfa':
    space = [
        {
            # "num_steps": tune.grid_search([5e6]),
            # "env.basic_only": tune.grid_search([1]),
            "num_steps": tune.grid_search([20e6]),
            "env.num_task_rooms": tune.grid_search([1]),
            "agent": tune.grid_search([
              'flat_cat_usfa'
              # 'object_usfa',
            ]),
            "seed": tune.grid_search([5]),
            "group": tune.grid_search(['cat_usfa-3']),
            "shared_ind_head": tune.grid_search([False]),
            'loss_fn': tune.grid_search(['qlearning', 'qlambda']),
            "sf_layers": tune.grid_search([[128, 128], [256, 256]]),
        },
    ]
  elif search == 'flat_usfa_dyna':
    space = [
        {
            # "num_steps": tune.grid_search([5e6]),
            # "env.basic_only": tune.grid_search([1]),
            "num_steps": tune.grid_search([20e6]),
            "env.num_task_rooms": tune.grid_search([1]),
            "agent": tune.grid_search(['flat_usfa_dyna',]),
            "seed": tune.grid_search([5]),
            "group": tune.grid_search(['flat_usfa_dyna-3']),
            "n_actions_dyna": tune.grid_search([5]),
            "model_coeff": tune.grid_search([.1, 1e-2]),
            "dyna_coeff": tune.grid_search([.1, .01, .001]),
            "task_weighted_model": tune.grid_search([False]),
        },
    ]
  elif search == 'usfa_dyna2':
    independent_heads={
      'task_weighted_dyna': tune.grid_search([False]),
      'task_weighted_model': tune.grid_search([False]),
      # 'weighted_coeff': tune.grid_search([1.0]),
      'sep_task_heads': tune.grid_search([True]),
      # 'model_discount': tune.grid_search([0.0]),
    }
    independent_heads_no_dyna={
      'task_weighted_dyna': tune.grid_search([False]),
      'task_weighted_model': tune.grid_search([False]),
      'sep_task_heads': tune.grid_search([True]),
      'dyna_coeff': tune.grid_search([0.0]),
      'model_coeff': tune.grid_search([0.0]),
    }
    shared_task_heads ={
      # 'task_weighted_dyna': tune.grid_search([True]),
      # 'task_weighted_model': tune.grid_search([True]),
      # 'weighted_coeff': tune.grid_search([1.0]),
      # 'unweighted_coeff': tune.grid_search([0.0]),
      # 'model_unweighted_coeff': tune.grid_search([0.0]),
      'sep_task_heads': tune.grid_search([False]),
      'sf_layers': tune.grid_search([[512, 512]]),
      'policy_layers': tune.grid_search([[128, 128]]),
      'policy_rep': tune.grid_search(['task']),
    }
    shared_task_heads_no_dyna ={
      'task_weighted_dyna': tune.grid_search([True]),
      'task_weighted_model': tune.grid_search([True]),
      # 'weighted_coeff': tune.grid_search([1.0]),
      # 'unweighted_coeff': tune.grid_search([0.0]),
      # 'model_unweighted_coeff': tune.grid_search([0.0]),
      'sep_task_heads': tune.grid_search([False]),
      'sf_layers': tune.grid_search([[256, 256]]),
      'policy_layers': tune.grid_search([[128, 128]]),
      'dyna_coeff': tune.grid_search([0.0]),
      'model_coeff': tune.grid_search([0.0]),
      'policy_rep': tune.grid_search(['task']),
    }
    shared_settings = {
      "num_steps": tune.grid_search([5e6]),
      "env.num_task_rooms": tune.grid_search([2]),
      'env.train_basic_objects': tune.grid_search([1]),
      "agent": tune.grid_search(["object_usfa_dyna"]),
      "seed": tune.grid_search([5]),
      "env.test_itermediary_rewards": tune.grid_search([False]),
      "env.evaluate_key_only": tune.grid_search([True]),
    }
    space = [
        # {
        #     **shared_settings,
        #     **shared_task_heads_no_dyna,
        #     "group": tune.grid_search(['usfa_dyna-56-shared-no-dyna']),
        #     'discount': tune.grid_search([.9, .99]),
        # },
        {
            **shared_settings,
            **shared_task_heads,
            "group": tune.grid_search(['usfa_dyna-68-shared']),
            "env.num_task_rooms": tune.grid_search([2]),
            "stop_dyna_state_grad": tune.grid_search([True]),
            "env.symbolic": tune.grid_search([True, False]),
            "backup_train_task": tune.grid_search([True, False]),
            "discount": tune.grid_search([.9, .99]),
            # "env.maze_idx": tune.grid_search([1]),
        },
        {
            **shared_settings,
            **independent_heads,
            "group": tune.grid_search(['usfa_dyna-68-ind']),
            "env.num_task_rooms": tune.grid_search([2]),
            "stop_dyna_state_grad": tune.grid_search([True]),
            "env.symbolic": tune.grid_search([False, True]),
            "backup_train_task": tune.grid_search([True, False]),
            "discount": tune.grid_search([.9, .99]),
        },
        # {
        #     **shared_settings,
        #     **independent_heads_no_dyna,
        #     "group": tune.grid_search(['usfa_dyna-56-ind-no-dyna']),
        #     'discount': tune.grid_search([.9, .99]),
        #     'sf_layers': tune.grid_search([[128]]),
        #     'unweighted_coeff': tune.grid_search([1.0, 0.0]),
        # },
    ]
  elif search == 'q':
    space = [
        {
            "num_steps": tune.grid_search([30e6]),
            "agent": tune.grid_search(['object_q', 'flat_q']),
            "seed": tune.grid_search([6]),
            "group": tune.grid_search(['q-2']),
            "env.basic_only": tune.grid_search([0]),
        },
    ]
  elif search == 'muzero':
    space = [
        {
            "num_steps": tune.grid_search([5e6]),
            "agent": tune.grid_search(['object_muzero', 'flat_muzero']),
            "seed": tune.grid_search([6]),
            "group": tune.grid_search(['muzero-3']),
            "mcts_train": tune.grid_search([True, False]),
            "env.basic_only": tune.grid_search([1]),
        },
    ]

  elif search == 'benchmark':
    space = [
        {
            "num_steps": tune.grid_search([30e6]),
            "agent": tune.grid_search([
              'object_muzero',
              'flat_muzero',
              'object_q',
              'flat_q',
              # 'object_usfa',
              # 'flat_usfa',
              ]),
            "seed": tune.grid_search([6]),
            "group": tune.grid_search(['benchmark-8-1room']),
            # "env.basic_only": tune.grid_search([0]),
            "env.num_task_rooms": tune.grid_search([1]),
            # "trace_length": tune.grid_search([20]),
            'env.test_itermediary_rewards': tune.grid_search([True]),
            # "out_conv_dim": tune.grid_search([16, 0]),
        },
        # {
        #     "num_steps": tune.grid_search([30e6]),
        #     "agent": tune.grid_search([
        #       'object_q']),
        #     "seed": tune.grid_search([6]),
        #     "group": tune.grid_search(['benchmark-4']),
        #     "env.basic_only": tune.grid_search([0]),
        #     "trace_length": tune.grid_search([40]),
        # },
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
