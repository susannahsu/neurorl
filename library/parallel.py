from typing import Optional, Union, List, Dict

from absl import flags
from absl import logging

import multiprocessing as mp
import os
import time
import datetime
import pickle 

from pathlib import Path
from ray import tune
import subprocess

from acme.utils import paths

import library.utils as utils

flags.DEFINE_integer('num_actors', 6, 'number of actors.')
flags.DEFINE_integer('config_idx', 1, 'number of actors.')
flags.DEFINE_integer('num_cpus', 16, 'number of cpus.')
flags.DEFINE_integer('memory', 120_000, 'memory (in mbs).')
flags.DEFINE_integer('max_concurrent', 12, 'number of concurrent jobs')
flags.DEFINE_string('account', '', 'account on slurm servers to use.')
flags.DEFINE_string('partition', 'kempner', 'account on slurm servers to use.')
flags.DEFINE_string('time', '0-06:00:00', '6 hours.')
flags.DEFINE_integer('num_gpus', 1, 'number of gpus.')
flags.DEFINE_bool('skip', False, 'whether to skip experiments that have already run.')
flags.DEFINE_bool('subprocess', False, 'label for whether this run is a subprocess.')
flags.DEFINE_bool('debug_parallel', False, 'whether to debug parallel runs.')

FLAGS = flags.FLAGS

def directory_not_empty(directory_path):
    return len(os.listdir(directory_path)) > 0

def date_time(time: bool=False):
  strkey = '%Y.%m.%d'
  if time:
    strkey += '-%H.%M'
  return datetime.datetime.now().strftime(strkey)

def gen_log_dir(
    base_dir="results/",
    date=False,
    hourminute=False,
    seed=None,
    return_kwpath=False,
    path_skip=[],
    **kwargs):

  kwpath = ','.join([f'{key[:4]}={value}' for key, value in kwargs.items() if not key in path_skip])

  if date:
    job_name = date_time(time=hourminute)
    path = Path(base_dir).joinpath(job_name).joinpath(kwpath)
  else:
    path = Path(base_dir).joinpath(kwpath)

  if seed is not None:
    path = path.joinpath(f'seed={seed}')

  if return_kwpath:
    return str(path), kwpath
  else:
    return str(path)

def make_program_command(
    folder: str,
    agent_config: str,
    env_config: str,
    wandb_init_kwargs: dict,
    filename: str = '',
    num_actors: int = 2,
    run_distributed: bool = False,
    **kwargs,
):
  wandb_project = wandb_init_kwargs['project']
  wandb_group = wandb_init_kwargs['group']
  wandb_name = wandb_init_kwargs['name']
  wandb_entity = wandb_init_kwargs['entity']

  assert filename, 'please provide file'
  str = f"""python {filename}
		--use_wandb=True
		--wandb_project='{wandb_project}'
		--wandb_entity='{wandb_entity}'
		--wandb_group='{wandb_group}'
    --wandb_name='{wandb_name}'
    --folder='{folder}'
    --agent_config='{agent_config}'
    --env_config='{env_config}'
    --num_actors={num_actors}
    --run_distributed={run_distributed}
    --parallel=False
  """
  for k, v in kwargs.items():
    str += f"--{k}={v}"
  return str


def create_and_run_ray_program(
    config,
    make_program_command,
    root_path: str = '.',
    folder : Optional[str] = None,
    wandb_init_kwargs: dict = None,
    default_env_kwargs: dict = None,
    skip: bool = True,
    debug: bool = False):
  """Create and run launchpad program
  """

  agent = config.get('agent', None)
  assert agent
  cuda = config.pop('cuda', None)

  # -----------------------
  # update env kwargs with config. HACK
  # -----------------------
  default_env_kwargs = default_env_kwargs or {}
  env_kwargs = dict()
  for key, value in default_env_kwargs.items():
    env_kwargs[key] = config.pop(key, value)

  # -----------------------
  # add env kwargs to path string. HACK
  # -----------------------
  # only use non-default arguments in path string
  env_path=dict()
  for k,v in env_kwargs.items():
    if v != default_env_kwargs[k]:
      env_path[k]=v

  # -----------------------
  # get log dir for experiment
  # -----------------------
  setting=dict(
    **env_path,
    **config
    )

  wandb_group = None
  if wandb_init_kwargs:
    wandb_group = wandb_init_kwargs.get('group', None)
  group = config.pop('group', wandb_group)

  # dir will be root_path/folder/group
  log_dir, exp_name = gen_log_dir(
    base_dir=os.path.join(root_path, folder, group),
    hourminute=False,
    return_kwpath=True,
    date=False,
    path_skip=['num_steps', 'num_learner_steps', 'group'],
    **log_path_config
    )


  print("="*50)
  if skip and os.path.exists(log_dir) and directory_not_empty(log_dir):
    print(f"SKIPPING\n{log_dir}")
    print("="*50)
    return
  else:
    print(f"RUNNING\n{log_dir}")
    print("="*50)

  # -----------------------
  # wandb settings
  # -----------------------
  if wandb_init_kwargs:
    wandb_init_kwargs['name']=exp_name # short display name for run
    if group is not None:
      wandb_init_kwargs['group']=group # short display name for run
    wandb_init_kwargs['dir'] = folder
  # needed for various services (wandb, etc.)
  os.chdir(root_path)

  # -----------------------
  # launch experiment
  # -----------------------

  if debug:
    config['num_steps'] = 50e3
  agent_config_file = f'{log_dir}/agent_config_kw.pkl'
  env_config_file = f'{log_dir}/env_config_kw.pkl'
  paths.process_path(log_dir)
  utils.save_config(agent_config_file, config)
  utils.save_config(env_config_file, env_kwargs)

  command = make_program_command(
    wandb_init_kwargs=wandb_init_kwargs,
    folder=log_dir,
    agent_config=agent_config_file,
    env_config=env_config_file,
  )
  print(command)
  command = command.replace("\n", '')
  cuda_env = os.environ.copy()
  if cuda:
    cuda_env["CUDA_VISIBLE_DEVICES"] = str(cuda)
  process = subprocess.Popen(command, env=cuda_env, shell=True)
  process.wait()


def run_ray(
    wandb_init_kwargs: dict,
    default_env_kwargs: dict,
    folder: str,
    space: Union[Dict, List[Dict]],
    use_wandb: bool = False,
    debug: bool = False,
    **kwargs):

  mp.set_start_method('spawn')
  root_path = str(Path().absolute())
  skip = FLAGS.skip


  def train_function(config):
    """Run inside threads and creates new process.
    """
    p = mp.Process(
      target=create_and_run_ray_program, 
      args=(config,),
      kwargs=dict(
        root_path=root_path,
        folder=folder,
        wandb_init_kwargs=wandb_init_kwargs if use_wandb else None,
        default_env_kwargs=default_env_kwargs,
        debug=debug,
        skip=skip,
        **kwargs)
      )
    p.start()
    wait_time = 30.0 # avoid collisions
    if wait_time and not debug:
      time.sleep(wait_time)
    p.join() # this blocks until the process terminates
    # this will call right away and end.

  if isinstance(space, dict):
    space = [space]

  from pprint import pprint
  pprint(space)

  experiment_specs = [tune.Experiment(
      name='experiment',
      run=train_function,
      config=s,
      resources_per_trial={"cpu": FLAGS.num_cpus, "gpu": FLAGS.num_gpus}, 
      local_dir='/tmp/ray',
    ) 
    for s in space
  ]
  tune.run_experiments(experiment_specs)

  import shutil
  if use_wandb:
    wandb_dir = wandb_init_kwargs.get("dir", './wandb')
    if os.path.exists(wandb_dir):
      shutil.rmtree(wandb_dir)

def get_all_configurations(spaces: Union[Dict, List[Dict]]):
    import itertools
    all_settings = []
    if isinstance(spaces, dict):
      spaces = [spaces]
    for space in spaces:
      # Extract keys and their corresponding lists from the space dictionary
      keys, value_lists = zip(*[(key, space[key]['grid_search']) for key in space])

      # Generate the Cartesian product of the value lists
      cartesian_product = itertools.product(*value_lists)

      # Create a list of dictionaries for each combination
      all_settings += [dict(zip(keys, values)) for values in cartesian_product]

    return all_settings

def get_agent_env_configs(
    config: dict,
    neither: List[str] = ['group', 'label'],
    default_env_kwargs: Optional[dict]=None):
  """
  Separate config into agent and env configs. Example below. Basically if key starts with "env.", it goes into an env_config.
  Example:
  config = {
    seed: 1,
    width: 2,
    env.room_size: 7,
    group: 'wandb_group4'
  }
  agent_config = {seed: 1, width: 2}
  env_config = {room_size: 7}
  """
  agent_config = dict()
  env_config = dict()

  for k, v in config.items():
    if 'env.' in k:
      # e.g. "env.room_size"
      env_config[k.replace("env.", "")] = v
    elif default_env_kwargs and k in default_env_kwargs:
      # e.g. "room_size"
      env_config[k] = v
    elif k in neither:
      pass
    else:
      agent_config[k] = v
  
  return agent_config, env_config

def run_sbatch(
    trainer_filename: str,
    wandb_init_kwargs: dict,
    folder: str,
    search_name: str,
    spaces: Union[Dict, List[Dict]],
    use_wandb: bool = False,
    num_actors: int = 4,
    debug: bool = False,
    run_distributed: bool = True):
  """For each possible configuration of a run, create a config entry. save a list of all config entries. When SBATCH is called, it will use the ${SLURM_ARRAY_TASK_ID} to run a particular one.
  """
  wandb_init_kwargs = wandb_init_kwargs or dict()
  #################################
  # create configs for all runs
  #################################
  root_path = str(Path().absolute())
  configurations = get_all_configurations(spaces=spaces)
  from pprint import pprint
  logging.info("searching:")
  pprint(configurations)
  save_configs = []
  for config in configurations:
    # either look for group name in setting, wandb_init_kwargs, or use search name
    if 'group' in config:
      group = config.pop('group')
    else:
      group = wandb_init_kwargs.get('group', search_name)

    agent_config, env_config = get_agent_env_configs(
        config=config)

    # dir will be root_path/folder/group/exp_name
    # exp_name is also name in wandb
    log_dir, exp_name = gen_log_dir(
      base_dir=os.path.join(root_path, folder, group),
      return_kwpath=True,
      path_skip=['num_steps', 'num_learner_steps', 'group'],
      **agent_config,
      **env_config,
      )

    save_config = dict(
      agent_config=agent_config,
      env_config=env_config,
      use_wandb=use_wandb,
      wandb_group=group,
      wandb_name=exp_name,
      folder=log_dir,
      num_actors=num_actors,
      run_distributed=run_distributed,
      wandb_project=wandb_init_kwargs.get('project', None),
      wandb_entity=wandb_init_kwargs.get('entity', None),
    )
    save_configs.append(save_config)

  #################################
  # save configs for all runs
  #################################
  # root_path/run_{search_name}-date-hour.pkl
  base_path = os.path.join(root_path, folder, 'runs', search_name)
  paths.process_path(base_path)

  base_filename = os.path.join(base_path, date_time(time=True))
  configs_file = f"{base_filename}_config.pkl"
  with open(configs_file, 'wb') as fp:
      pickle.dump(save_configs, fp)
      logging.info(f'Saved: {configs_file}')

  #################################
  # create run.sh file to run with sbatch
  #################################
  python_file_contents = f"python {trainer_filename}"
  python_file_contents += f" --config_file={configs_file}"
  python_file_contents += f" --use_wandb={use_wandb}"
  python_file_contents += f" --num_actors={num_actors}"
  if debug:
    python_file_contents += f" --config_idx=1"
  else:
    python_file_contents += f" --config_idx=$SLURM_ARRAY_TASK_ID"
  python_file_contents += f" --run_distributed={run_distributed}"
  python_file_contents += f" --subprocess={True}"
  python_file_contents += f" --make_path={False}"

  run_file = f"{base_filename}_run.sh"

  if debug:
    # create file and run single python command
    run_file_contents = "#!/bin/bash\n" + python_file_contents
    logging.warning("only running first config")
    print(run_file_contents)
    with open(run_file, 'w') as file:
      # Write the string to the file
      file.write(run_file_contents)
    process = subprocess.Popen(f"chmod +x {run_file}", shell=True)
    process = subprocess.Popen(run_file, shell=True)
    process.wait()
    return

  #################################
  # create sbatch file
  #################################
  sbatch_contents = f"#SBATCH --gres=gpu:{FLAGS.num_gpus}\n"
  sbatch_contents += f"#SBATCH -c {FLAGS.num_cpus}\n"
  sbatch_contents += f"#SBATCH --mem {FLAGS.memory}\n"
  # sbatch_contents += f"#SBATCH --mem-per-cpu={FLAGS.memory}\n"
  sbatch_contents += f"#SBATCH -p {FLAGS.partition}\n"
  sbatch_contents += f"#SBATCH -t {FLAGS.time}"
  sbatch_contents += f"#SBATCH --account {FLAGS.account}\n"
  sbatch_contents += f"#SBATCH -o {base_filename}_id=%j.out\n"
  sbatch_contents += f"#SBATCH -e {base_filename}_id=%j.err\n"

  run_file_contents = "#!/bin/bash\n" + sbatch_contents + python_file_contents
  print("-"*20)
  print(run_file_contents)
  print("-"*20)
  with open(run_file, 'w') as file:
    # Write the string to the file
    file.write(run_file_contents)

  total_jobs = len(save_configs)
  max_concurrent = min(FLAGS.max_concurrent, total_jobs)
  sbatch_command = f"sbatch --array=1-{total_jobs}%{max_concurrent} {run_file}"
  logging.info(sbatch_command)
  process = subprocess.Popen(f"chmod +x {run_file}", shell=True)
  process = subprocess.Popen(sbatch_command, shell=True)
  process.wait()

