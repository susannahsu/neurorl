# DEBUGGING, single stream
python -m ipdb -c continue configs/biorl_trainer.py \
  --search='qlearning' \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --use_wandb=False \
  --wandb_entity=wcarvalho92 \
  --wandb_project=neurorl_debug

# DEBUGGING, no_jit, single stream
JAX_DISABLE_JIT=1 python -m ipdb -c continue configs/biorl_trainer.py \
  --search='qlearning' \
  --parallel='none' \
  --run_distributed=False \
  --debug=True \
  --use_wandb=False \
  --wandb_entity=wcarvalho92 \
  --wandb_project=neurorl_debug


# running a search, single-stream
python configs/biorl_trainer.py \
  --search='qlearning' \
  --parallel='sbatch' \
  --run_distributed=False \
  --use_wandb=True \
  --partition=kempner \
  --account=kempner_fellows \
  --wandb_entity=wcarvalho92 \
  --wandb_project=neurorl

