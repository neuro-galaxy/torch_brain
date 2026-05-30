#!/bin/bash
#SBATCH --job-name=single_gpu_mila
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100l
#SBATCH --mem=48G
#SBATCH --partition=long
#
#set -e
#dataset=allen_brain_observatory_calcium
dataset=allen_brain_observatory_calcium
WANDB_PROJECT=allen_bo_calcium

module load anaconda/3
module load cuda/11.2/nccl/2.8
module load cuda/11.2
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA

#conda activate poyo
source $HOME/poyo_env/bin/activate

# wandb credentials
set -a
source .env
set +a

cd /home/mila/x/xuejing.pan/POYO/project-kirby

# Uncompress the data to SLURM_TMPDIR single node
snakemake --rerun-triggers=mtime -c1 allen_brain_observatory_calcium_unfreeze
nvidia-smi

srun --export=ALL,WANDB_PROJECT=allen_bo_calcium,HYDRA_FULL_ERROR=1,CUDA_LAUNCH_BLOCKING=1,TORCH_USE_CUDA_DSA=1 python /home/mila/x/xuejing.pan/POYO/project-kirby/examples/capoyo/train_all.py \
        --config-name train_allen_bo.yaml data_root=$SLURM_TMPDIR/uncompressed epochs=2000 name=allen_brain_observatory_calcium_all_one_gpu