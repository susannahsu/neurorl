# FAS install

## load modules
```
module load cuda/11.3.1-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01
module load gcc/9.5.0-fasrc01
```


find `cudnn_version.h `. Mine was at 
```
cudnn_version_path=/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda11-fasrc01/include/cudnn_version.h
```
cat $cudnn_version_path | grep CUDNN_MAJOR -A 2

Make sure `LD_LIBRARY_PATH` properly links by setting
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cudnn/lib
# mine
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda11-fasrc01/lib/
```

# Install
Change `LD_LIBRARY_PATH` in `install-fas.sh` as needed based on above
```
chmod u+x install-fas.sh
./install-fas.sh
```

**how does install work?**

[TODO]

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

Experiments are defined by configs. To make your own experiment, copy one of the configs (e.g. [biorl_trainer.py](configs/biorl_trainer.py)). You will need to change two functions:
1. `make_environment`: this function specifies how environments are constructed. This codebase assumes `dm_env` environments so make sure to convert `gym` environments to `dm_env`.
2. `setup_experiment_inputs`: this function specifies how agents are loaded. In the example given, a q-learning agent is loaded.

Agents are defined with 3 things (e.g. [biorl_trainer.py](configs/biorl_trainer.py#L124)):
1. a config ([example](td_agents/q_learning.py#L27)), which specified default values
2. a builder ([example](td_agents/q_learning.py#L30)), which specifies how the learner/replay buffer/actor will be created. you mainly change this object in order to change something about learning.
3. a network_factory ([example](td_agents/q_learning.py#L11)), which creates the neural networks that define the agnet.




# Available (Recurrent) Agents

1. [Q-learning](td_agents/q_learning.py)
2. [Successor Features](td_agents/sf_agents.py)
<!-- 2. Object-oriented Successor Features -->