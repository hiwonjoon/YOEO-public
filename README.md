# You Only Evaluate Once (YOEO)

## Prerequisite

```
mujoco200
conda
```

- Remember to add the mujoco directory to `LD_LIBRARY_PATH` environment variable.
- All the other dependencies will be handled in the following installation script via conda.

## Install

```
git clone <anonymized for dual blind>
cd <cloned-dir>
conda create --file env.yaml --name YOEO
conda activate YOEO
# in the case of error during creation, use conda update commands:
# conda env update --file env.yaml
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
```

## Run

### Train Y
    ```
    python -m YOEO.scripts.value_training --log_dir ./log/directory/you/want --seed {seed you want} --config_file ./experiments/Y.gin ./experiments/envs/{hopper,walker2d,etc}.gin --config_params env_id=\'{d4rl env id, like hopper-medium-replay-v0}\'
    ```

### Train YOEO (and ablations)
  - YOEO
    ```
    python -m YOEO.scripts.offline_rl --log_dir ./log/directory/you/want --seed 1 --config_file ./experiments/ValidQbeta.gin ./experiments/envs/{hopper,walker2d,etc}.gin --config_params env_id=\'{d4rl env id, like hopper-medium-replay-v0}\' Y_chkpt=\'./path/to/trained/Y/value.tf\' lambda={0.1 or 1.0}
    ```

  - $Q^\beta$ without regularizations
    ```
    python -m YOEO.scripts.offline_rl --log_dir ./log/directory/you/want --seed 1 --config_file ./experiments/Qbeta.gin ./experiments/envs/{hopper,walker2d,etc}.gin --config_params env_id=\'{d4rl env id, like hopper-medium-replay-v0}\'
    ```

  - For other ablation studies, you can just change the gin file, or you can provide the hyperparameters using `--config_params` arguments, then the provided arguments will overide the arguments in gin file.

### Observe training statistics & results
  - Use Tensorboard
    ```
    tensorboard --logdir <log directory>
    ```
