
# Jan 4, 2024

- I have added `contrastive_dyna` as an RL agent. Here we learn a k-step transition- and reward- model with contrastive learning. The loss function learns from both real data and simulated data.
- I have added examples learning with these agents to `catch_trainer` and `minigrid_trainer`.
- I did some refactoring of `muzero` to better use model-based pieces with `dyna`.

# Jan 3, 2024

I changed the install instructions for the FAS clusters. The main changes are:

- no longer load `gcc` module
- changed install to rely on `cuda 11.8.0` was previous `cuda 11.3.0`. It looks like some jax-based libraries no longer supports `cuda < 11.8.0`?
- directly load more strict package settings when creating conda environment
- swap order of installing jax/ACME. now: first JAX, then ACME.
- changed ACME install to rely on **latest** acme commit. inspecting history of commits, seems relatively stable.
  - also changed instructions for setting install path
- added manually pinning `dm-haiku==0.0.10` in addition to `chex==0.1.6`
- changed test instruction to use `catch_trainer.py` which is a simple trainer for this library