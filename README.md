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
Run `bash install-fas.sh` changing `LD_LIBRARY_PATH` on `line 41` as needed.


## Running experiments

Set up environment:
```
module load cuda/11.3.1-fasrc01  
module load cudnn/8.9.2.26_cuda11-fasrc01
module load gcc/9.5.0-fasrc01
```

Set the results directory
```
export RL_RESULTS_DIR=results_dir
# optionally add to conda activation
echo 'export RL_RESULTS_DIR=results_dir' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```