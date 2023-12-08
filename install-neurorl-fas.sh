# salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1

# NOTES: cuda/11.3.1 seems needed for jax==0.4.3
# NOTES: jax==0.4.3 seems needed for acme
module load cuda/11.3.1-fasrc01  
module load cudnn/8.9.2.26_cuda11-fasrc01
module load gcc/9.5.0-fasrc01

mamba create --name neurorl python=3.9 pip wheel -y

source activate neurorl

conda env update --name neurorl --file conda_env.yaml

#############################################
# ACME
#############################################
git clone https://github.com/google-deepmind/acme.git _acme
echo "Installing ACME manually. At the time, acme from pip is missing some features"
cd _acme
git checkout 4525ade7015c46f33556e18d76a8d542b916f264
pip install --editable ".[jax,testing,tf,envs]" 
cd ..

#############################################
# JAX
#############################################
# inspired from: https://github.com/wcarvalho/oo-model/blob/fixing/install.sh
# need to pin this value for ACME
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda11-fasrc01/lib/
pip install --upgrade "jax[cuda11_pip]==0.4.3" "jaxlib==0.4.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install chex==0.1.6
pip install gym[accept-rom-license]

#############################################
# Setup activate/deactivate with correct PYTHONPATH and LD_LIBRARY_PATH
#############################################
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

#############################################
# test install
#############################################
python -c "import jax; jax.random.split(jax.random.PRNGKey(42), 2); print('hello world'); print(f'GPUS: {jax.device_count()}')"

python _acme/examples/baselines/rl_discrete/run_r2d2.py
