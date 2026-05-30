#!/bin/bash
#SBATCH --job-name=multi_gpu_mila
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus=24
#SBATCH --gres=gpu:4
#SBATCH --mem=64GB
#SBATCH --partition=short-unkillable

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
snakemake --rerun-triggers=mtime --config tmp_dir=$SLURM_TMPDIR -c1 $rule

# Important info for parallel GPU processing
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export NCCL_BLOCKING_WAIT=1

echo $MASTER_ADDR:$MASTER_PORT

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
    name=multi_gpu_mila \
    base_lr=1e-5 \
    precision=16 \
    num_workers=6 \
    model=poyo_single_session \
    nodes=1 \
    gpus=4