# Install

## Setting up conda environment
**Activate interactive terminal**
```
salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1
```

**load modules**
```
module load python/3.10.12-fasrc01
module load cuda/11.8.0-fasrc01
```

**Create and activate conda environment**
```
mamba create -n neurorl python=3.9 pip wheel -y
mamba env update -f conda_env.yml

source activate neurorl
```

**test that using python 3.9**
```
python -c "import sys; print(sys.version)"
```
if not, your `path` environment variable/conda environment activation might be giving another system `python` priority.

**Setting up `LD_LIBRARY_PATH`**.
This is important for jax to properly link to cuda. Unfortunately, relatively manual. You'll need to find where your `cudnn` lib is. Mine is at the path below. `find` might be a useful bash command for this.

```
# put cudnn path at beginning of path (before $LD_LIBRARY_PATH)
export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda11-fasrc01/lib/:$LD_LIBRARY_PATH

# add conda lib to end of path (after $LD_LIBRARY_PATH)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
```

## Installing JAX

**pip install:**.
```
pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Expected errors:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
chex 0.1.85 requires jax>=0.4.16, but you have jax 0.4.3 which is incompatible.
flax 0.7.5 requires jax>=0.4.19, but you have jax 0.4.3 which is incompatible.
orbax-checkpoint 0.4.1 requires jax>=0.4.9, but you have jax 0.4.3 which is incompatible.
```

**test jax install**
```
TF_CPP_MIN_LOG_LEVEL=0 python -c "import jax; print(f'GPUS: {jax.device_count()}'); jax.random.split(jax.random.PRNGKey(42), 2); print('hello world');"
```


## Installing ACME
Installing ACME manually. ACME from pip has not been updated in a bit. But checkpoint here seems relatively stable.
Install ACME in a directory specified by `$INSTALL_LOC`. Change as needed.
```
# store current directory
cur_dir=`pwd`

# make install directory
export INSTALL_LOC=$HOME/installs
mkdir -p $INSTALL_LOC

# install acme
acme_loc=$INSTALL_LOC/acme
git clone https://github.com/google-deepmind/acme.git $acme_loc
cd $acme_loc
git checkout 1177501df180edadd9f125cf5ee960f74bff64af

# this will take a while... (30 min-1hr or so for me)
pip install --editable ".[jax,testing,tf,envs]"

# return to original directory
cd $cur_dir
```
Expected error:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
minigrid 2.3.1 requires pygame>=2.4.0, but you have pygame 2.1.0 which is incompatible.
```

### install some older package versions
```
pip install dm-haiku==0.0.10
```

### test whether can run experiment
```
# reset LD_LIBRARY PATH
unset LD_LIBRARY
unset PYTHONPATH

# load env
source activate neurorl

# load modules
module load python/3.10.12-fasrc01
module load cuda/11.8.0-fasrc01

# set environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export PYTHONPATH=$PYTHONPATH:.

# run experiment
python configs/catch_trainer.py \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --use_wandb=False \
  --search='baselines'
```
