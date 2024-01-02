
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

pip install --upgrade "jax[cuda11_pip]==0.4.3" "jaxlib==0.4.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# older versions needed for ACME/minigrid :(
pip install chex==0.1.6

```


# test install
**test jax install**
```
TF_CPP_MIN_LOG_LEVEL=1 python -c "import jax; print(f'GPUS: {jax.device_count()}'); jax.random.split(jax.random.PRNGKey(42), 2); print('hello world'); "
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


# Setup activate/deactivate with correct PYTHONPATH and LD_LIBRARY_PATH

```
activation_dir=$CONDA_PREFIX/etc/conda/activate.d
mkdir -p $activation_dir
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo 'module load cuda/11.3.1-fasrc01  ' > $activation_dir/env_vars.sh
echo 'module load cudnn/8.9.2.26_cuda11-fasrc01' >> $activation_dir/env_vars.sh
echo 'module load gcc/9.5.0-fasrc01' >> $activation_dir/env_vars.sh
echo 'export PYTHONPATH=$PYTHONPATH:.' >> $activation_dir/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> $activation_dir/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda11-fasrc01/lib/' >> $activation_dir/env_vars.sh
echo 'unset LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```
