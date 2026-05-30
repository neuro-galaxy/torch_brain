#!/bin/bash
#SBATCH --job-name=single_gpu_mila
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#
#set -e


module load anaconda/3
module load cuda/11.2/nccl/2.8
module load cuda/11.2

conda activate poyo

srun python /home/mila/x/xuejing.pan/POYO/project-kirby/allenBO_download.py