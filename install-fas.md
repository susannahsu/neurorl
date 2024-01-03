
Note:

- you need to get into an interactive terminal
- enterring an interactive terminal resets your LD_LIBRARY_PATH and loaded modules.

---
---
# Install

## Setting up install environment
**Activate interactive terminal**
```
salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1
```

**NOTES**:

- cuda/11.3.1 seems needed for jax==0.4.3
- jax==0.4.3 seems needed for acme

```
module load cuda/11.3.1-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01
module load gcc/9.5.0-fasrc01
```

**Create and activate conda environment**
```
mamba create --name neurorl python=3.9 pip wheel -y
source activate neurorl
mamba env update --name neurorl --file conda_env.yaml
```

## Installing  ACME
Installing ACME manually. At the time, acme from pip was missing some features.
Will install ACME in a directory specified by `$INSTALL_LOC`. Change as needed
```
cur_dir=`pwd`
export INSTALL_LOC=$HOME/installs
mkdir -p $INSTALL_LOC
acme_loc=$INSTALL_LOC/acme
git clone https://github.com/google-deepmind/acme.git $acme_loc
cd $acme_loc
git checkout 4525ade7015c46f33556e18d76a8d542b916f264
pip install --editable ".[jax,testing,tf,envs]" 
cd $cur_dir
```

## Installing JAX

**Setup `LD_LIBRARY_PATH`**. This is important for jax to properly link to cuda. Unfortunately, still relatively manual. You'll need to find where your `cudnn` lib is. Mine is at the path below. `find` might be a useful bash command for this.
```
export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda11-fasrc01/lib/:$LD_LIBRARY_PATH
```

**NOTE:** need to pin this value for ACME

```
# older versions needed for ACME/minigrid :(
pip install distrax==0.1.4
pip install chex==0.1.6

pip install --upgrade "jax==0.4.3" "jaxlib==0.4.3+cuda11.cudnn86" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


```


# test install
**test jax install**
```
TF_CPP_MIN_LOG_LEVEL=0 python -c "import jax; print(f'GPUS: {jax.device_count()}'); jax.random.split(jax.random.PRNGKey(42), 2); print('hello world'); "
```

**test this lib**. make sure conda environment is active! first setup `PYTHONPATH` to include current directory and `LD_LIBRARY_PATH` to point to CONDA environment, in addition to `cudnn` environment.
```
export PYTHONPATH=$PYTHONPATH:.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH

python configs/catch_trainer.py \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --use_wandb=False \
  --search='baselines'
```


# Setup conda activate/deactivate

**ONE TIME CHANGE TO MAKE YOUR LIFE EASIER**. if you want to avoid having to load modules and set environment variables each time you load this environment, you can add loading things to the activation file. Below is how.

```
# first activate env
source activate neurorl

# make activation/deactivation directories
activation_dir=$CONDA_PREFIX/etc/conda/activate.d
mkdir -p $activation_dir
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# module loading added to activation
echo 'module load cuda/11.3.1-fasrc01  ' > $activation_dir/env_vars.sh
echo 'module load cudnn/8.9.2.26_cuda11-fasrc01' >> $activation_dir/env_vars.sh

# setting PYTHONPATH added to activation
echo 'export PYTHONPATH=$PYTHONPATH:.' >> $activation_dir/env_vars.sh

# setting LD_LIBRARY_PATH added to activation
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> $activation_dir/env_vars.sh

# undoing LD_LIBRARY_PATH added to deactivation
echo 'unset LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
``````

For automatically setting the LD library path to point to the current cudnn directory do the following
```
# general
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cudnn/lib' >> $activation_dir/env_vars.sh
# mine
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda11-fasrc01/lib/' >> $activation_dir/env_vars.sh
```