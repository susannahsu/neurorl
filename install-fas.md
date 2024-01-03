# Install

## Setting up install environment
**Activate interactive terminal**
```
salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1
```

**load modules**
```
module load python/3.10.12-fasrc01
module load cuda/11.8.0-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01
```

**Create and activate conda environment**
```
mamba env create -f conda-env.yml
source activate neurorl
```


## Installing  ACME
Installing ACME manually. At the time, acme from pip was missing some features.
Will install ACME in a directory specified by `$INSTALL_LOC`. Change as needed
```
# store current directory
cur_dir=`pwd`

# make install directory
export INSTALL_LOC=$HOME/installs
mkdir -p $INSTALL_LOC

# install acme
acme_loc=$INSTALL_LOC/acme
git clone https://github.com/google-deepmind/acme.git $acme_loc
git checkout 1177501df180edadd9f125cf5ee960f74bff64af
cd $acme_loc
pip install --upgrade --editable ".[jax,testing,tf,envs]"

# return to original directory
cd $cur_dir
```
Expected errors:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
minigrid 2.3.1 requires pygame>=2.4.0, but you have pygame 2.1.0 which is incompatible.
```
## Installing JAX

**Setup `LD_LIBRARY_PATH`**. This is important for jax to properly link to cuda. Unfortunately, still relatively manual. You'll need to find where your `cudnn` lib is. Mine is at the path below. `find` might be a useful bash command for this.
```
export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda11-fasrc01/lib/:$LD_LIBRARY_PATH
```

### pip install

```
pip install --upgrade "jax[cuda11_local]" "jaxlib==0.4.3+cuda11.cudnn86" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Expected errors:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
chex 0.1.85 requires jax>=0.4.16, but you have jax 0.4.3 which is incompatible.
flax 0.7.5 requires jax>=0.4.19, but you have jax 0.4.3 which is incompatible.
orbax-checkpoint 0.4.1 requires jax>=0.4.9, but you have jax 0.4.3 which is incompatible.
```

### **INSTALL older packages versions** (needed for acme)
```
pip install chex==0.1.6
pip install dm-haiku==0.0.10
```

Expected errors:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
distrax 0.1.4 requires chex>=0.1.8, but you have chex 0.1.6 which is incompatible.
```

# test install

### test jax install
```
TF_CPP_MIN_LOG_LEVEL=0 python -c "import jax; print(f'GPUS: {jax.device_count()}'); jax.random.split(jax.random.PRNGKey(42), 2); print('hello world'); "
```

### test this lib
make sure conda environment is active! first setup `PYTHONPATH` to include current directory and `LD_LIBRARY_PATH` to point to CONDA environment, in addition to `cudnn` environment.
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