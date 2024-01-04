# Install

[FAS Install and Setup](install-fas.md)

## Setup conda activate/deactivate

**ONE TIME CHANGE TO MAKE YOUR LIFE EASIER**. if you want to avoid having to load modules and set environment variables each time you load this environment, you can add loading things to the activation file. Below is how.

```
# first activate env
source activate neurorl

# make activation/deactivation directories
activation_dir=$CONDA_PREFIX/etc/conda/activate.d
mkdir -p $activation_dir
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# module loading added to activation
echo 'module load cuda/11.8.0-fasrc01  ' > $activation_dir/env_vars.sh
echo 'module load cudnn/8.9.2.26_cuda11-fasrc01' >> $activation_dir/env_vars.sh

# setting PYTHONPATH added to activation
echo 'export PYTHONPATH=$PYTHONPATH:.' >> $activation_dir/env_vars.sh

# setting LD_LIBRARY_PATH added to activation
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> $activation_dir/env_vars.sh

# undoing LD_LIBRARY_PATH added to deactivation
echo 'unset LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```

For automatically setting the LD library path to point to the current cudnn directory do the following
```
# general
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cudnn/lib' >> $activation_dir/env_vars.sh
# mine
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda11-fasrc01/lib/' >> $activation_dir/env_vars.sh
```



## (Optionally) permanently set the results directory
```
echo 'export RL_RESULTS_DIR=${results_dir}' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
example:
```
echo 'export RL_RESULTS_DIR=$HOME/rl_results' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Otherwise, can set each time run experiment
```
RL_RESULTS_DIR=${results_dir} python trainer.py
```

## (Optional) setup wandb
```
pip install wandb
wandb login
```
Once you set up a wandb project and have logged runs, group them by the following settings:
- Group: name of search run
- Name: name of individual runs (this aggregates all seeds together)

# Running experiments

load environment
```
source activate neurorl 
```

**how do experiments work?**

Experiments are defined by configs. To make your own experiment, copy one of the configs (e.g. [catch_trainer.py](configs/catch_trainer.py)). You will need to change two functions:
1. `make_environment`: this function specifies how environments are constructed. This codebase assumes `dm_env` environments so make sure to convert `gym` environments to `dm_env`.
2. `setup_experiment_inputs`: this function specifies how agents are loaded. In the example given, a q-learning agent is loaded.

Agents are defined with 3 things (e.g. [catch_trainer.py](configs/catch_trainer.py#L124)):
1. a config ([example](td_agents/q_learning.py#L27)), which specified default values
2. a builder ([example](td_agents/q_learning.py#L30)), which specifies how the learner/replay buffer/actor will be created. you mainly change this object in order to change something about learning.
3. a network_factory ([example](td_agents/q_learning.py#L11)), which creates the neural networks that define the agnet.




# Available (Recurrent) Agents

1. Model-free: [Q-learning](td_agents/q_learning.py)
2. Model-free: [Successor Features](td_agents/usfa.py)
3. Model-based: [Dyna w. Contrastive World Model](td_agents/contrastive_dyna.py)
3. Model-based: [MuZero](td_agents/muzero.py)

### example trainers
1. simplest: [catch trainer](configs/catch_trainer.py)
2. [minigrid trainer](configs/minigrid_trainer.py)

Each one shows how to run Q-learning, dyna, or MuZero.