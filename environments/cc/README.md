# Running on Narval

Running on Narval requires one to recreate an appropriate environment. 

## Installing via conda and pip

A standard route is to install `conda` and `pip`. This requires internet access and has to be done on a login node. You can use `build_env_cc.sh` for that. Typically, `requirements_cc.txt` will be out-of-date, so you will need to recreate from its current state using `requirements.txt` as a template. A typical set of steps might include:

* Removing torch and pydantic as dependencies (handled by conda)
* Removing the pandas version qualifier (conflicts with compute-canada-built packages)

Then, you have to *both* load the right Python module and activate the right conda environment, e.g.:

```
source ~/.bashrc  # To make conda available
load module python/3.9.6
conda activate poyo
```

A symptom of not doing the latter is e.g. getting glibc errors when loading some libraries, e.g. h5py. To run the allen sdk, you will in addition need to `module load postgresql`.

After that, you can pip install the module via regular means, i.e. `pip install -e .`.

## Pre-packaged environment

[Conda pack](https://conda.github.io/conda-pack/) is an interesting alternative that could be used, eventually, to build an environment on the mila cluster and ship it to Narval. We have not confirmed that it works, however.

## Using offline wandb

If you're seeing this network error: `Network error (ConnectionError), entering retry loop`, 
this might mean that your network is slow or blocking the connection to the wandb servers.

You can run the training in offline mode by setting the `WANDB_mode` environment variable to `offline`:
```bash
WANDB_mode=offline CUDA_VISIBLE_DEVICES=0 python train.py --config-name train_mc_maze_small log_dir=./logs
```

Be sure to specify the `log_dir` argument to save the wandb logs where you can access them 
later (i.e. in a directory that is not deleted after the training is done, or a node
is terminated).

In CC, this directory is scratch.

The following can be run from the login node or any node that has access to the logs 
directory and to a stable internet connection:

In stdout, you can identify the wandb run ID, which you will use to sync the run. If you
have a log file you can do the following:
```bash
grep "Wandb ID: " log_file.log
```
which should output something like:
```bash
Wandb ID: 2tfkbhxk
```

You can then use this to find the run folder in the logs directory (replace `./logs/` with the path to the logs directory):
```bash
ls ./logs/wandb/ | grep "2tfkbhxk" 
```
which should output something like:
```bash
offline-run-20240426_135211-2tfkbhxk
```

Note that if you didn't get the wandb ID, you can use the timestamp to find the run folder.

If your run is done, you can sync the run with the following command:
```bash
wandb sync ./logs/wandb/offline-run-20240426_135211-2tfkbhxk
```

But you probably want to track your training, which you can do by scheduling a sync every 5 minutes:
```bash
watch -n 300 wandb sync ./logs/wandb/offline-run-20240426_135211-2tfkbhxk
```

The run should now be visible in the wandb dashboard `https://wandb.ai/neuro-galaxy/poyo/runs/2tfkbhxk`
