#!/bin/bash
#SBATCH --job-name=single_gpu_mila
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --partition=unkillable

dataset=perich_single_session
rule=perich_miller_unfreeze

module load anaconda/3
module load cuda/11.2/nccl/2.8
module load cuda/11.2

conda activate poyo

# wandb credentials
set -a
source .env
set +a

# Uncompress the data to SLURM_TMPDIR
snakemake --rerun-triggers=mtime --config -c1 $rule

nvidia-smi

# Run experiments
pwd
which python
srun python train.py \
    data_root=$SLURM_TMPDIR/uncompressed/ \
    train_datasets=$dataset \
    val_datasets=$dataset \
    eval_epochs=1  \
    epochs=1 \
    pct_start=0.9 \
    batch_size=128 \
    name=single_gpu_mila \
    base_lr=1e-5 \
    precision=16 \
    num_workers=6 \
    model=poyo_single_session \
    nodes=1 \
    gpus=1