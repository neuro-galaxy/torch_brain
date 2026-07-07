#!/bin/bash
#SBATCH --job-name=possm-perich-repro
#SBATCH --array=0-13
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/possm_perich_repro/slurm/%A_%a.out
set -euo pipefail

DATASETS=(
  perich_co_c_20161013
  perich_co_c_20161021
  perich_co_t_20130819
  perich_co_t_20130821
  perich_co_t_20130823
  perich_co_t_20130903
  perich_co_t_20130905
  perich_co_t_20130909
  perich_rt_t_20130820
  perich_rt_t_20130822
  perich_rt_t_20130830
  perich_rt_t_20130904
  perich_rt_t_20130906
  perich_rt_t_20130910
)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
DATA_ROOT="${DATA_ROOT:-$HOME/scratch/data/processed-brainsets}"
LOG_ROOT="${LOG_ROOT:-$REPO/logs/possm_perich_repro}"

mkdir -p "$LOG_ROOT/slurm"
export MPLCONFIGDIR=/tmp/matplotlib-possm-perich-repro
cd "$REPO/examples/possm"

"$PY" train.py --config-name=train_poyo_mp \
  dataset="$DATASET" \
  data_root="$DATA_ROOT" \
  log_dir="$LOG_ROOT/$DATASET" \
  epochs=1000 \
  batch_size=128 \
  eval_batch_size=128 \
  eval_epochs=1 \
  num_workers=6 \
  gpus=1 \
  wandb.enable=false
