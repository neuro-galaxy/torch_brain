# POSSM example

This directory contains the POSSM training and finetuning scripts used for
decoding experiments on Brainsets datasets.

The reproduction setup in this directory is the 50 ms POSSM-GRU 9M configuration
used for the Perich-Miller single-session checks:

- C center-out: `c_20161013`, `c_20161021`
- T center-out: `t_20130819`, `t_20130821`, `t_20130823`, `t_20130903`,
  `t_20130905`, `t_20130909`
- T random-target: `t_20130820`, `t_20130822`, `t_20130830`, `t_20130904`,
  `t_20130906`, `t_20130910`

Expected reference metrics:

```text
C-CO 2016: 0.9549 ± 0.0012
T-CO:      0.8863 ± 0.0222
T-RT:      0.7687 ± 0.0669
```

## Environment

From the repository root, create an environment and install the package:

```bash
uv venv ~/scratch/torch-brain-venv
source ~/scratch/torch-brain-venv/bin/activate
uv pip install -r examples/possm/requirements-possm.txt
uv pip install -e ".[dev]" --no-deps
```

`requirements-possm.txt` is the pinned package set from the environment used for
the reproduction runs; the editable `torch-brain` install is done separately by
the second command.

The reproduced runs used `torch==2.11.0` with `torch.version.cuda == "13.0"`.

## Prepare Brainsets data

Prepare the Perich-Miller dataset with Brainsets:

```bash
brainsets prepare perich_miller_population_2018
```

The training scripts read processed Brainsets files from `data_root`. On a
cluster, it is usually better to point this at scratch storage, for example:

```bash
DATA_ROOT=$HOME/scratch/data/processed-brainsets
```

If your Brainsets installation writes to a different location, pass that path as
`data_root=...` on the Hydra command line.

## Train one session locally

Run from this directory:

```bash
cd examples/possm

python train.py --config-name=train_poyo_mp \
  dataset=perich_co_c_20161013 \
  data_root=$HOME/scratch/data/processed-brainsets \
  log_dir=$HOME/scratch/logs/possm/perich_co_c_20161013 \
  wandb.enable=false
```

`train_poyo_mp` uses the POSSM-9M GRU model, 50 ms bins, 1000 epochs, batch size
128, and UnitDropout.

Useful overrides:

```bash
epochs=100
batch_size=64
eval_batch_size=64
num_workers=8
gpus=1
wandb.enable=false
```

## Reproduce the 14 Perich single-session results on SLURM

The helper script runs the 14 sessions listed above as a SLURM array:

```bash
mkdir -p logs/possm_perich_repro/slurm

PY=$HOME/scratch/torch-brain-venv/bin/python \
DATA_ROOT=$HOME/scratch/data/processed-brainsets \
sbatch examples/possm/slurm_possm_perich_repro.sh
```

Edit the `#SBATCH` lines in `slurm_possm_perich_repro.sh` for your cluster
partition, GPU type, memory, and wall-clock limit.
The `#SBATCH --output` path is where the metric table is printed; SLURM does not
expand shell variables in that line, so edit it directly if you want logs outside
`logs/possm_perich_repro/slurm`.

Collect results after the array finishes:

```bash
python examples/possm/collect_perich_repro_results.py <job_id> \
  --log-root logs/possm_perich_repro/slurm
```

The task ids are grouped as:

- `0-1`: C center-out
- `2-7`: T center-out
- `8-13`: T random-target

## Dataset configs

The 14 single-session configs live under `configs/dataset/`:

- `perich_co_c_*.yaml`
- `perich_co_t_*.yaml`
- `perich_rt_t_*.yaml`

They use the standard Brainsets train/valid/test masks. They do not override the
split with a chronological `sampling_intervals_modifier`.

The key metric detail is `eval_interval`:

- center-out sessions evaluate R2 on `movement_phases.reach_period`
- random-target sessions evaluate R2 on `movement_phases.random_period`

To be precise, for center-out tasks, metrics are computed only on the reach
period; for random-target tasks, metrics are computed only on the random period.

## Finetuning

Finetuning uses `configs/finetune.yaml`:

```bash
python finetune.py ckpt_path=/path/to/checkpoint.ckpt \
  dataset=perich_co_c_20161013 \
  data_root=$HOME/scratch/data/processed-brainsets \
  wandb.enable=false
```

Checkpoints are loaded with `weights_only=False` because POSSM checkpoints carry
vocabulary metadata that PyTorch's safe weights-only loading path cannot unpickle.
