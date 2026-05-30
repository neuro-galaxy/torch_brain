#!/bin/bash
#SBATCH --job-name=multi_gpu_mila
#SBATCH --output=/home/mila/x/xuejing.pan/POYO/project-kirby/examples/capoyo/slurm_outputs/slurm_output_%j.txt
#SBATCH --error=/home/mila/x/xuejing.pan/POYO/project-kirby/examples/capoyo/slurm_outputs/slurm_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --time=32:00:00
#SBATCH --partition=main

export WANDB_PROJECT=allen_bo_calcium

module load miniconda/3
module load cuda/11.2/nccl/2.8
module load cuda/11.2

#conda activate poyo
conda activate poyo_conda
# wandb credentials
set -a
source .env
set +a

cd /home/mila/x/xuejing.pan/POYO/project-kirby/
# Uncompress the data to SLURM_TMPDIR
#snakemake --rerun-triggers=mtime --config tmp_dir=$SLURM_TMPDIR -c4 allen_brain_observatory_calcium_unfreeze
snakemake --rerun-triggers=mtime -c1 allen_brain_observatory_calcium_unfreeze
# Important info for parallel GPU processing
#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
#export MASTER_ADDR=$(hostname)
#export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
#export NCCL_BLOCKING_WAIT=1

#echo $MASTER_ADDR:$MASTER_PORT

nvidia-smi

# Run experiments
pwd
which python

cd /home/mila/x/xuejing.pan/POYO/project-kirby/examples/capoyo/poyo_hparam_sweep

#for runs
#srun --export=ALL,WANDB_PROJECT=allen_bo_calcium python train.py \
#       --config-name train_allen_bo.yaml data_root=$SLURM_TMPDIR/uncompressed gpus=2 epochs=1000

#for sweeps
wandb agent neuro-galaxy/allen_bo_calcium/d63jo2y5

#CUDA_VISIBLE_DEVICES=0 wandb agent neuro-galaxy/allen_bo_calcium/hcrbvrqe 
#CUDA_VISIBLE_DEVICES=1 wandb agent neuro-galaxy/allen_bo_calcium/hcrbvrqe