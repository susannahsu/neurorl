"""
Running experiments:
--------------------

# DEBUGGING, single stream
python -m ipdb -c continue configs/imagination_trainer.py \
  --search='initial' \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --use_wandb=False \
  --wandb_entity=wcarvalho92 \
  --wandb_project=imagination_debug

JAX_DISABLE_JIT=1 python -m ipdb -c continue configs/imagination_trainer.py \
  --search='initial' \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --use_wandb=False \
  --wandb_entity=wcarvalho92 \
  --wandb_project=imagination_debug


# DEBUGGING, parallel
python -m ipdb -c continue configs/imagination_trainer.py \
  --search='initial' \
  --parallel='sbatch' \
  --debug_parallel=True \
  --run_distributed=False \
  --use_wandb=True \
  --wandb_entity=wcarvalho92 \
  --wandb_project=imagination_debug \


# running, parallel
python configs/imagination_trainer.py \
  --search='initial' \
  --parallel='sbatch' \
  --run_distributed=True \
  --use_wandb=True \
  --partition=kempner \
  --account=kempner_fellows \
  --wandb_entity=wcarvalho92 \
  --wandb_project=imagination

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

from acme.jax import networks as networks_lib
from acme.jax.networks import duelling
from acme import wrappers as acme_wrappers
from acme import specs
from acme.jax import experiments
import gymnasium
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp

from td_agents import q_learning, basics, muzero
from lib import muzero_mlps

from lib.dm_env_wrappers import GymWrapper
import lib.env_wrappers as env_wrappers
import lib.experiment_builder as experiment_builder
import lib.parallel as parallel
import lib.utils as utils
import lib.networks as networks

from envs import mental_blocks

flags.DEFINE_string('config_file', '', 'config file')
flags.DEFINE_string('search', 'default', 'which search to use.')
flags.DEFINE_string(
    'parallel', 'none', "none: run 1 experiment. sbatch: run many experiments with SBATCH. ray: run many experiments with say. use sbatch with SLUM or ray otherwise.")
flags.DEFINE_bool(
    'debug', False, 'If in debugging mode, only 1st config is run.')
flags.DEFINE_bool(
    'make_path', True, 'Create a path under `FLAGS>folder` for the experiment')

FLAGS = flags.FLAGS


State = jax.Array

def observation_encoder(
    inputs: acme_wrappers.observation_action_reward.OAR,
    num_actions: int):
  """Dummy function that just concatenations obs, action, reward.
  If there's a batch dim, applies vmap first."""
  def fn(x):
    x = jnp.concatenate((
      x.observation.reshape(-1),  #[N*D]
      jnp.expand_dims(jnp.tanh(x.reward), 0),   # [1]
      jax.nn.one_hot(x.action, num_actions)))  # [A]
    return hk.Linear(512)(x)

  has_batch_dim = inputs.reward.ndim > 0
  if has_batch_dim:
    # have batch dimension
    fn = jax.vmap(fn)
  return fn(inputs)


def make_qlearning_networks(
        env_spec: specs.EnvironmentSpec,
        config: q_learning.Config,
        ):
  """Builds default R2D2 networks for Atari games."""

  num_actions = int(env_spec.actions.maximum - env_spec.actions.minimum)

  def make_core_module() -> q_learning.R2D2Arch:

    observation_fn = functools.partial(
      observation_encoder, num_actions=num_actions)
    return q_learning.R2D2Arch(
      torso=hk.to_module(observation_fn)('obs_fn'),
      memory=networks.DummyRNN(),  # nothing happens
      head=duelling.DuellingMLP(num_actions,
                                hidden_sizes=[config.q_dim]))

  return networks_lib.make_unrollable_network(
    env_spec, make_core_module)


def make_muzero_networks(
    env_spec: specs.EnvironmentSpec,
    config: muzero.Config,
    **kwargs) -> muzero.MuZeroNetworks:
  """Builds default MuZero networks for BabyAI tasks."""

  num_actions = int(env_spec.actions.maximum - env_spec.actions.minimum)

  def make_core_module() -> muzero.MuZeroNetworks:

    ###########################
    # Setup observation and state functions
    ###########################
    observation_fn = functools.partial(observation_encoder,
                                       num_actions=num_actions)
    observation_fn = hk.to_module(observation_fn)('obs_fn')
    state_fn = networks.DummyRNN()

    ###########################
    # Setup transition function: ResNet
    ###########################
    def transition_fn(action: int, state: State):
      action_onehot = jax.nn.one_hot(
          action, num_classes=num_actions)
      assert action_onehot.ndim in (1, 2), "should be [A] or [B, A]"

      def _transition_fn(action_onehot, state):
        """ResNet transition model that scales gradient."""
        # action: [A]
        # state: [D]
        out = muzero_mlps.SimpleTransition(
            num_blocks=config.transition_blocks)(
            action_onehot, state)
        out = muzero.scale_gradient(out, config.scale_grad)
        return out, out

      if action_onehot.ndim == 2:
        _transition_fn = jax.vmap(_transition_fn)
      return _transition_fn(action_onehot, state)
    transition_fn = hk.to_module(transition_fn)('transition_fn')

    ###########################
    # Setup prediction functions: policy, value, reward
    ###########################
    root_value_fn = hk.nets.MLP(
        (128, 32, config.num_bins), name='pred_root_value')
    root_policy_fn = hk.nets.MLP(
        (128, 32, num_actions), name='pred_root_policy')
    model_reward_fn = hk.nets.MLP(
        (32, 32, config.num_bins), name='pred_model_reward')

    if config.seperate_model_nets:
      # what is typically done
      model_value_fn = hk.nets.MLP(
          (128, 32, config.num_bins), name='pred_model_value')
      model_policy_fn = hk.nets.MLP(
          (128, 32, num_actions), name='pred_model_policy')
    else:
      model_value_fn = root_value_fn
      model_policy_fn = root_policy_fn

    def root_predictor(state: State):
      assert state.ndim in (1, 2), "should be [D] or [B, D]"

      def _root_predictor(state: State):
        policy_logits = root_policy_fn(state)
        value_logits = root_value_fn(state)

        return muzero.RootOutput(
            state=state,
            value_logits=value_logits,
            policy_logits=policy_logits,
        )
      if state.ndim == 2:
        _root_predictor = jax.vmap(_root_predictor)
      return _root_predictor(state)

    def model_predictor(state: State):
      assert state.ndim in (1, 2), "should be [D] or [B, D]"

      def _model_predictor(state: State):
        reward_logits = model_reward_fn(state)

        policy_logits = model_policy_fn(state)
        value_logits = model_value_fn(state)

        return muzero.ModelOutput(
            new_state=state,
            value_logits=value_logits,
            policy_logits=policy_logits,
            reward_logits=reward_logits,
        )
      if state.ndim == 2:
        _model_predictor = jax.vmap(_model_predictor)
      return _model_predictor(state)

    return muzero.MuZeroArch(
        observation_fn=observation_fn,
        state_fn=state_fn,
        transition_fn=transition_fn,
        root_pred_fn=root_predictor,
        model_pred_fn=model_predictor)

  return muzero.make_network(
    environment_spec=env_spec,
    make_core_module=make_core_module,
    **kwargs)


def make_environment(seed: int,
                     difficulty: int= 1,
                     evaluation: bool = False,
                     **kwargs) -> dm_env.Environment:
  """Loads environments. 
  """
  del seed
  del evaluation

  _, input_stacks, goal_stacks = mental_blocks.create_random_problem(difficulty=difficulty)

  # create simulator
  environment = mental_blocks.Simulator(input_stacks=input_stacks, goal_stacks=goal_stacks)
  
  # dm_env
  env =  mental_blocks.EnvWrapper(environment)

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
    config = q_learning.Config(**config_kwargs)
    builder = basics.Builder(
      config=config,
      LossFn=q_learning.R2D2LossFn(
          discount=config.discount,
          importance_sampling_exponent=config.importance_sampling_exponent,
          burn_in_length=config.burn_in_length,
          max_replay_size=config.max_replay_size,
          max_priority_weight=config.max_priority_weight,
          bootstrap_n=config.bootstrap_n,
      ))
    network_factory = functools.partial(make_qlearning_networks, config=config)
  elif agent == 'muzero':
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
    network_factory = functools.partial(make_muzero_networks, config=config)
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


  config = experiment_config_inputs.agent_config
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
      samples_per_insert=1.0,
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
  if search == 'initial':
    space = [
        {
            "group": tune.grid_search(['run-2']),
            "agent": tune.grid_search(['qlearning', 'muzero']),
            "seed": tune.grid_search([1]),
            "env.difficulty": tune.grid_search([7]),
        }
    ]
  elif search == 'muzero':
    space = [
        {
            "agent": tune.grid_search(['muzero']),
            "seed": tune.grid_search([1]),
            "seed": tune.grid_search([1]),
            "env.difficulty": tune.grid_search([7]),
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
