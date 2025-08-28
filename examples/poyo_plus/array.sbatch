#!/bin/bash
#SBATCH --job-name=poyo-ibl-array
#SBATCH --account=bdnx-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA40x4-preempt
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:5:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --requeue
#SBATCH --signal=B:TERM@300
#SBATCH --nice=10000

set -euo pipefail
mkdir -p logs

# ------------------- user knobs -------------------
CSV="${CSV:-experiments.csv}"   # header: session_id,region
SCRIPT=examples/poyo_plus/finetune.py
BASE_CFG="--config-name train_poyo_plus_ibl_finetune"
BASE_OVERRIDES="data_root=/work/hdd/bdnx/mazabou/processed/ batch_size=64 gpus=1 num_workers=2 nodes=1 optim.base_lr=5e-5 eval_epochs=2 optim.lr_decay_start=0.9 epochs=40"
CKPT="/u/mazabou/torch_brain/poyo-plus-ibl-bs4x-baselr6.25e-6-iblsampler-epoch=951.ckpt"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export HYDRA_FULL_ERROR=1
# --------------------------------------------------

# ----------- env setup (modules + venv) -----------
module purge
module load gcc/11.4.0 openmpi/4.1.6 cuda/11.8.0 cue-login-env/1.1 slurm-env/0.1 default-s11

# Activate venv and resolve absolute python path
source /u/mazabou/venv/bin/activate
PYTHON="$(command -v python)"    # now inside venv
if [[ -z "${PYTHON:-}" || ! -x "$PYTHON" ]]; then
  echo "ERROR: Python interpreter not found after venv activation." >&2
  which python || true
  which python3 || true
  exit 1
fi

echo "[INFO] Using Python: $PYTHON"
"$PYTHON" -V || true
# --------------------------------------------------

# Map array index -> CSV row (skip header)
LINE_NO=$(( SLURM_ARRAY_TASK_ID + 2 ))
IFS=, read -r session_id region < <(sed -n "${LINE_NO}p" "$CSV")

if [[ -z "${session_id:-}" || -z "${region:-}" ]]; then
  echo "Empty row for task $SLURM_ARRAY_TASK_ID (line $LINE_NO). Exiting."
  exit 0
fi

run_name="poyo-plus-ibl-finetune_${region}_${session_id}_stimulus_side"
echo "[$(date)] Start task=$SLURM_ARRAY_TASK_ID session_id=$session_id region=$region"

# Graceful preemption
trap 'echo "[$(date)] Preemption signal; exiting so Slurm requeues."; exit 0' TERM

# IMPORTANT: quote ckpt_path value for Hydra (filename contains '=')
# Build Hydra-safe overrides
OV_SESS="dataset.0.selection.0.sessions=['$session_id']"

srun --export=ALL --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus=1 \
  "$PYTHON" "$SCRIPT" $BASE_CFG \
  $BASE_OVERRIDES \
  wandb.run_name="$run_name" \
  ckpt_path="'$CKPT'" \
  "$OV_SESS" \
  "train_transforms.0.region=$region" \
  "eval_transforms.0.region=$region"


echo "[$(date)] Done task=$SLURM_ARRAY_TASK_ID session_id=$session_id region=$region"
