# salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1

# NOTES: cuda/11.3.1 seems needed for jax==0.4.3
# NOTES: jax==0.4.3 seems needed for acme
module load cuda/11.3.1-fasrc01  
module load cudnn/8.9.2.26_cuda11-fasrc01
module load gcc/9.5.0-fasrc01

mamba create --name humansf python=3.9 pip wheel -y

source activate humansf

#############################################
# Minigrid
# at the time, needed this version because the babyai bot was not available via pip
#############################################

git clone https://github.com/Farama-Foundation/Minigrid .minigrid
cd .minigrid
git checkout e726259e86d555c7055fb48bd5842cf37af78bfd
pip install --editable .
cd ..


#############################################
# ACME
#############################################
git clone https://github.com/google-deepmind/acme.git .acme
cd .acme
git checkout 4525ade7015c46f33556e18d76a8d542b916f264
pip install --editable ".[jax,testing,tf,envs]" 
cd ..

#############################################
# JAX
#############################################
# inspired from: https://github.com/wcarvalho/oo-model/blob/fixing/install.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda11-fasrc01/lib/
pip install --upgrade "jax[cuda11_pip]==0.4.3" "jaxlib==0.4.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

#############################################
# test install
#############################################
python -c "import jax; jax.random.split(jax.random.PRNGKey(42), 2); print('hello world'); print(f'GPUS: {jax.device_count()}')"

# inspired from: https://github.com/wcarvalho/oo-model/blob/fixing/Makefile
CONDA_ENV_DIR=${HOME}/.conda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CONDA_ENV_DIR}/envs/humansf/lib/

python .acme/examples/baselines/rl_discrete/run_r2d2.py

# git add install-fas.sh .gitignore; git add -u; git commit -m "added installation files