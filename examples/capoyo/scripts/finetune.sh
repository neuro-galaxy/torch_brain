#!/bin/bash
#SBATCH --job-name=single_gpu_mila
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --time=48:00:00
#SBATCH --reservation=ubuntu2204
#
#set -e
dataset=openscope_calcium

module load anaconda/3
module load cuda/11.2/nccl/2.8
module load cuda/11.2

conda activate poyo

# wandb credentials
set -a
source .env
set +a

# Uncompress the data to SLURM_TMPDIR single ndoe
#snakemake --forceall --rerun-triggers=mtime -c1 openscope_calcium_unfreeze
snakemake --rerun-triggers=mtime -c1 openscope_calcium_unfreeze
nvidia-smi

#For multi-session
srun python train.py \
        data_root=$SLURM_TMPDIR/uncompressed/ \
        train_datasets=$dataset \
        val_datasets=$dataset \
        name=FINETUNE_ind_transfer_soma \
        epochs=1000 \
        ckpt_path=/home/mila/x/xuejing.pan/POYO/project-kirby/logs/lightning_logs/8u7542ov/last.ckpt 
