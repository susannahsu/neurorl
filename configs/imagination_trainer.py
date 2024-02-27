"""
Running experiments:
--------------------

# DEBUGGING, single stream
python -m ipdb -c continue configs/imagination_trainer.py \
  --search='initial' \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --use_wandb=True \
  --wandb_entity=wcarvalho92 \
  --wandb_project=imagination_debug

# DEBUGGING, turn off JIT
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


# launching jobs on slurm
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
from typing import Dict

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
import numpy as np
import rlax
import wandb
import matplotlib.pyplot as plt

from td_agents import q_learning, basics, muzero
from library import muzero_mlps

from library.dm_env_wrappers import GymWrapper
#import library.env_wrappers as env_wrappers
import library.experiment_builder as experiment_builder
import library.parallel as parallel
import library.utils as utils
import library.networks as networks

from envs import mental_blocks
from envs import mental_blocks_cfg

flags.DEFINE_string('config_file', '', 'config file') # ('flag name', 'default value', 'value interpretation')
flags.DEFINE_string('search', 'default', 'which search to use.')
flags.DEFINE_string(
    'parallel', 'none', "none: run 1 experiment. sbatch: run many experiments with SBATCH. ray: run many experiments with say. use sbatch with SLUM or ray otherwise.")
flags.DEFINE_bool(
    'debug', False, 'If in debugging mode, only 1st config is run.')
flags.DEFINE_bool(
    'make_path', True, 'Create a path under `FLAGS>folder` for the experiment')

FLAGS = flags.FLAGS
# more flags are in parallel.py


State = jax.Array


@dataclasses.dataclass
class QConfig(q_learning.Config):
  q_dim: int = 512 
  state_dim: int = 512

"""
def observation_encoder(
    inputs: acme_wrappers.observation_action_reward.OAR,
    num_actions: int):
  #Dummy function that just concatenations obs, action, reward.
  #If there's a batch dim, applies vmap first.
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
"""

def observation_encoder(
    inputs: acme_wrappers.observation_action_reward.OAR,
    num_actions: int,
    max_num_stacks: int=mental_blocks_cfg.MAX_STACKS,
    max_num_blocks: int=mental_blocks_cfg.MAX_BLOCKS):
  # create embeddings for different elements in state repr
  cur_stack_embed = hk.Linear(32, w_init=hk.initializers.TruncatedNormal())
  table_stack_embed = hk.Linear(32, w_init=hk.initializers.TruncatedNormal())
  goal_stack_embed = hk.Linear(32, w_init=hk.initializers.TruncatedNormal())
  intersection_embed = hk.Linear(16, w_init=hk.initializers.TruncatedNormal())
  cur_pointer_embed = hk.Linear(16, w_init=hk.initializers.TruncatedNormal())
  table_pointer_embed = hk.Linear(16, w_init=hk.initializers.TruncatedNormal())
  input_parsed_embed = hk.Linear(8, w_init=hk.initializers.TruncatedNormal())
  goal_parsed_embed = hk.Linear(8, w_init=hk.initializers.TruncatedNormal())
  # embeddings for prev reward and action
  reward_embed = hk.Linear(16, w_init=hk.initializers.RandomNormal())
  action_embed = hk.Linear(16, w_init=hk.initializers.TruncatedNormal())
  # backbone of the encoder: mlp with relu
  mlp = hk.nets.MLP([256,256,256], activate_final=False) # default RELU activations between layers (and after final layer)
  def fn(x, dropout_rate=None):
    # concatenate embeddings and previous reward and action
    x = jnp.concatenate((
        cur_stack_embed(x.observation[:max_num_stacks*max_num_blocks, :].reshape(-1)),
        table_stack_embed(x.observation[max_num_stacks*max_num_blocks:max_num_stacks*max_num_blocks+max_num_blocks, :].reshape(-1)),
        goal_stack_embed(x.observation[max_num_stacks*max_num_blocks+max_num_blocks:2*max_num_stacks*max_num_blocks+max_num_blocks, :].reshape(-1)),
        intersection_embed(x.observation[2*max_num_stacks*max_num_blocks+max_num_blocks:2*max_num_stacks*max_num_blocks+max_num_blocks+max_num_stacks, :].reshape(-1)),
        cur_pointer_embed(x.observation[-4, :].reshape(-1)),
        table_pointer_embed(x.observation[-3, :].reshape(-1)),
        input_parsed_embed(x.observation[-2, :].reshape(-1)),
        goal_parsed_embed(x.observation[-1, :].reshape(-1)),
        # jnp.expand_dims(x.reward, 0),  # [1]
        # jax.nn.one_hot(x.action, num_actions), # [A]
        reward_embed(jnp.expand_dims(x.reward, 0)), 
        action_embed(jax.nn.one_hot(x.action, num_actions))  
      ))
    # relu first, then mlp, relu
    x = jax.nn.relu(x)
    x = mlp(x, dropout_rate=dropout_rate)
    return jax.nn.relu(x)
  # If there's a batch dim, applies vmap first.
  has_batch_dim = inputs.reward.ndim > 0
  if has_batch_dim: # have batch dimension
    fn = jax.vmap(fn)
  return fn(inputs)


def make_qlearning_networks(
        env_spec: specs.EnvironmentSpec,
        config: q_learning.Config,
        ):
  """Builds default R2D2 networks for Atari games."""

  num_actions = int(env_spec.actions.maximum - env_spec.actions.minimum) + 1

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

  num_actions = int(env_spec.actions.maximum - env_spec.actions.minimum) + 1

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
  
class QObserver(basics.ActorObserver):
  def __init__(self,
               period=5000,
               prefix: str = 'QObserver'):
    super(QObserver, self).__init__()
    self.period = period
    self.prefix = prefix
    self.idx = -1
    self.logging = True
    self.action_dict = {0:"next_stack", 1:"previous_stack",
                        2:"next_table", 3:"previous_table",
                        4:"remove", 5:"add",
                        6:"parse_input", 7:"parse_goal"}

  def wandb_log(self, d: dict):
    if self.logging:
      if wandb.run is not None:
        wandb.log(d)
      else:
        self.logging = False
        self.period = np.inf

  def observe_first(self, state: basics.ActorState, timestep: dm_env.TimeStep) -> None:
    """Observes the initial state and initial time-step.

    Usually state will be all zeros and time-step will be output of reset."""
    self.idx += 1

    # epsiode just ended, flush metrics if you want
    if self.idx > 0:
      self.get_metrics()

    # start collecting metrics again
    self.actor_states = [state]
    self.timesteps = [timestep]
    self.actions = []

  def observe_action(self, state: basics.ActorState, action: jax.Array) -> None:
    """Observe state and action that are due to observation of time-step.

    Should be state after previous time-step along"""
    self.actor_states.append(state)
    self.actions.append(action)

  def observe_timestep(self, timestep: dm_env.TimeStep) -> None:
    """Observe next.

    Should be time-step after selecting action"""
    self.timesteps.append(timestep)

  def get_metrics(self, max_steps:int = 100) -> Dict[str, any]:
    """Returns metrics collected for the current episode."""
    if not self.idx % self.period == 0:
      return
    if not self.logging:
      return
    ##################################
    # successor features
    ##################################
    print('\nlogging!')
    # first prediction is empty (None)
    results = {}
    q_values = [s.predictions for s in self.actor_states[1:]]
    q_values = jnp.stack(q_values)
    npreds = len(q_values)
    actions = jnp.stack(self.actions)[:npreds]
    q_values = rlax.batched_index(q_values, actions)
    action_names = [self.action_dict[a.item()] for a in actions]
    rewards = jnp.stack([t.reward for t in self.timesteps[1:]])
    observations = jnp.stack([t.observation.observation for t in self.timesteps[1:]])
    # log the metrics
    results["actions"] = actions
    results["action_names"] = action_names
    results["q_values"] = q_values
    results["rewards"] = rewards
    results["observations"] = observations 
    # plot reward vs q value pred
    fig, ax = plt.subplots()
    ax.plot(q_values, label='q_values')
    ax.plot(rewards, label='rewards')
    ax.set_xlabel('step')
    total_reward = rewards.sum()
    ax.set_title(f"Total reward:\nR={total_reward}")
    ax.legend()
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, max_steps)
    self.wandb_log({f"{self.prefix}/reward_prediction": wandb.Image(fig)}) # Log the plot to wandb
    plt.close(fig) # Close the plot
    # plot each state action in the episode
    fig, ax = plt.subplots(max_steps//10, 10, figsize=(50,9*(max_steps//10))) # 10 plots a row
    stateh = observations[0].shape[0]
    statew = observations[0].shape[1]
    extent = (0, statew, stateh, 0)
    for t in range(npreds):
      irow = t//10
      jcol = t%10
      ax[irow,jcol].imshow(observations[t], cmap="binary", vmin=0, vmax=1, extent=extent)
      ax[irow,jcol].grid(color='gray', linewidth=2)
      ax[irow,jcol].set_xticks(np.arange(statew))
      ax[irow,jcol].set_yticks(np.arange(stateh))
      ax[irow,jcol].set_title(f"A={action_names[t]}\nR={rewards[t]}\nQ={q_values[t]}")
    self.wandb_log({f"{self.prefix}/trajectory": wandb.Image(fig)})
    plt.close(fig)

    episode_reward = jnp.sum(rewards)
    print('episode rewards', episode_reward)
    episode_reward_threshold = mental_blocks_cfg.EPISODE_REWARD_THRESHOLD
    if mental_blocks_cfg.CURRICULUM==None: 
      print("no dynamic curriculum, use fixed distribution", mental_blocks_cfg.CURRICULUM)
      return # no dynamic curriculum
    elif episode_reward > episode_reward_threshold and mental_blocks_cfg.CURRICULUM==0: 
      print("continuing final stabalizing ", mental_blocks_cfg.CURRICULUM)
      mental_blocks_cfg.EPISODE_REWARD_THRESHOLD = 2.3
      return # final stabalizing
    elif episode_reward < episode_reward_threshold and mental_blocks_cfg.CURRICULUM==2: 
      print("continuing easiest curriculum ", mental_blocks_cfg.CURRICULUM)
      mental_blocks_cfg.EPISODE_REWARD_THRESHOLD = 1.4
      return # continue level 2 training 
    elif episode_reward < episode_reward_threshold and mental_blocks_cfg.CURRICULUM==0: 
      mental_blocks_cfg.CURRICULUM = mental_blocks_cfg.MAX_BLOCKS # fall back to highest curriculum level
      print("stop stabalizing, fall back to the highest curriculum ", mental_blocks_cfg.CURRICULUM)
      mental_blocks_cfg.EPISODE_REWARD_THRESHOLD = 2.3
    elif episode_reward < episode_reward_threshold and mental_blocks_cfg.CURRICULUM>2: 
      mental_blocks_cfg.CURRICULUM -= 1 # return to prev curriculum level
      print("curriculum decreased to ", mental_blocks_cfg.CURRICULUM)
      mental_blocks_cfg.EPISODE_REWARD_THRESHOLD = 2 if mental_blocks_cfg.CURRICULUM >= 3 else 1.4
    elif episode_reward > episode_reward_threshold and mental_blocks_cfg.CURRICULUM < mental_blocks_cfg.MAX_BLOCKS:
      mental_blocks_cfg.CURRICULUM += 1 # increase curriculum level
      print("curriculum increased to ", mental_blocks_cfg.CURRICULUM)
      mental_blocks_cfg.EPISODE_REWARD_THRESHOLD = 2 if mental_blocks_cfg.CURRICULUM == 3 else 2.3
    elif episode_reward > episode_reward_threshold and mental_blocks_cfg.CURRICULUM == mental_blocks_cfg.MAX_BLOCKS:
      mental_blocks_cfg.CURRICULUM = 0 # set to wide distribution to stabalize training
      print("start final stabalizing, curriculum set to ", mental_blocks_cfg.CURRICULUM)
      mental_blocks_cfg.EPISODE_REWARD_THRESHOLD = 2.3
    print("episode reward threshold ", mental_blocks_cfg.EPISODE_REWARD_THRESHOLD)

    return results
  

class MuObserver(QObserver):
  def __init__(self,
               period=5000,
               prefix: str = 'MuObserver'):
    super(MuObserver, self).__init__()
    self.period = period
    self.prefix = prefix
  def get_metrics(self, max_steps:int = 100) -> Dict[str, any]:
    """Returns metrics collected for the current episode."""
    if not self.idx % self.period == 0:
      return
    if not self.logging:
      return
    print('\nlogging!')
    # first prediction is empty (None)
    npreds = len(self.actor_states[1:])
    # print(npreds, self.actor_states[1:])
    actions = jnp.stack(self.actions)[:npreds]
    action_names = [self.action_dict[a.item()] for a in actions]
    rewards = jnp.stack([t.reward for t in self.timesteps[1:]])
    observations = jnp.stack([t.observation.observation for t in self.timesteps[1:]])
    # plot each state action in the episode
    fig, ax = plt.subplots(max_steps//10, 10, figsize=(50,9*(max_steps//10))) # 10 plots a row
    stateh = observations[0].shape[0]
    statew = observations[0].shape[1]
    extent = (0, statew, stateh, 0)
    for t in range(npreds):
      irow = t//10
      jcol = t%10
      ax[irow,jcol].imshow(observations[t], cmap="binary", vmin=0, vmax=1, extent=extent)
      ax[irow,jcol].grid(color='gray', linewidth=2)
      ax[irow,jcol].set_xticks(np.arange(statew))
      ax[irow,jcol].set_yticks(np.arange(stateh))
      ax[irow,jcol].set_title(f"A={action_names[t]}\nR={rewards[t]}")
    self.wandb_log({f"{self.prefix}/trajectory": wandb.Image(fig)})
    plt.close(fig)

def make_environment(seed: int ,
                     difficulty=None, # None or int {2,3,...,MAX_BLOCKS}
                     evaluation: bool = False,
                     action_cost: float = 1e-3,
                     max_steps: int = 100,
                     **kwargs) -> dm_env.Environment:
  """Loads environments. 
  """
  del seed
  del evaluation
  
  # create dummy problem to initialize the env obj
  _, input_stacks, goal_stacks = mental_blocks.create_random_problem(difficulty=difficulty, 
                                                                    cur_curriculum=mental_blocks_cfg.CURRICULUM)

  # create simulator
  environment = mental_blocks.Simulator(input_stacks=input_stacks, goal_stacks=goal_stacks, 
                                        action_cost=action_cost, max_steps=max_steps,
                                        difficulty=difficulty)
  
  rng = np.random.default_rng(1)
  
  # dm_env
  env =  mental_blocks.EnvWrapper(environment, rng)

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
      ActorCls=functools.partial(
        basics.BasicActor,
        observers=[QObserver(period=1 if debug else 5000)],
        ),
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
      ActorCls=functools.partial(
        basics.BasicActor,
        observers=[MuObserver(period=1 if debug else 10000)],
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
          #reanalyze_ratio=config.reanalyze_ratio,
          reanalyze_ratio=0.25, # how frequently to tune value loss wrt tree node values (faster but less accurate)
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

  if FLAGS.make_path: # default False from parallel slurm jobs
    # i.e. ${folder}/runs/${date_time}/
    folder = parallel.gen_log_dir(
        base_dir=os.path.join(folder, 'rl_results'),
        hourminute=True,
        date=True,
    )

  ########################
  # override with config settings, e.g. from parallel run
  ########################
  if FLAGS.config_file: # parallel run should pass in a config_file that's created online/temporarily
    configs = utils.load_config(FLAGS.config_file)
    config = configs[FLAGS.config_idx-1]  # FLAGS.config_idx starts at 1 with SLURM
    logging.info(f'loaded config: {str(config)}')

    agent_config_kwargs.update(config['agent_config'])
    env_kwargs.update(config['env_config'])
    folder = config['folder']

    num_actors = config['num_actors'] # default 6 from parallel slurm jobs
    run_distributed = config['run_distributed'] # default True from parallel slurm jobs

    wandb_init_kwargs['group'] = config['wandb_group']
    wandb_init_kwargs['name'] = config['wandb_name']
    wandb_init_kwargs['project'] = config['wandb_project']
    wandb_init_kwargs['entity'] = config['wandb_entity']

    if not config['use_wandb']:
      wandb_init_kwargs = dict()


  if FLAGS.debug and not FLAGS.subprocess: # FLAGS.subprocess default True from parallel slurm jobs
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
  wandb_init_kwargs = setup_wandb_init_kwargs() # group will be 'FLAGS.search' by default

  folder = FLAGS.folder or os.environ.get('RL_RESULTS_DIR', None)
  if not folder:
    folder = '/tmp/rl_results_dir'

  assert FLAGS.debug is False, 'only run debug if not running many things in parallel'
  # and FLAGS.parallel should be 'none' for debug
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
  elif FLAGS.parallel == 'sbatch': # fasrc is sbatch system
    # this will submit multiple sbatch jobs, each will call run_single(distributed=True)
    parallel.run_sbatch(
      trainer_filename=__file__,
      wandb_init_kwargs=wandb_init_kwargs,
      use_wandb=FLAGS.use_wandb,
      folder=folder,
      run_distributed=FLAGS.run_distributed, # usually user command will set this to True if parallel
      search_name=FLAGS.search,
      debug=FLAGS.debug_parallel,
      spaces=sweep(FLAGS.search), # usually search is 'initial'
      num_actors=FLAGS.num_actors) # default flag is 6 (in parallel.py)

def sweep(search: str = 'default'):
  if search == 'initial':
    space = [
        {
            "group": tune.grid_search(['38Q']),
            "num_steps": tune.grid_search([50e6]),
            "env.action_cost": tune.grid_search([1e-2]),
            "max_grad_norm": tune.grid_search([80.0]),
            "learning_rate": tune.grid_search([1e-4]),
            "epsilon_begin": tune.grid_search([0.9]),
            # "seed": tune.grid_search([1]), 

            # "agent": tune.grid_search(['muzero']),
            # "num_bins": tune.grid_search([1001]),  # for muzero
            # "num_simulations": tune.grid_search([5]), # for muzero

            "agent": tune.grid_search(['qlearning']),
            "state_dim": tune.grid_search([512]),
            "q_dim": tune.grid_search([512]),
            # "env.difficulty": tune.grid_search([6]),
        }
    ]
  elif search == 'muzero':
    space = [
        {
            "agent": tune.grid_search(['muzero']),
            "num_steps": tune.grid_search([50e6]),
            "seed": tune.grid_search([1]),
            # "seed": tune.grid_search([1]),
            "env.difficulty": tune.grid_search([2,3,7]),
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
