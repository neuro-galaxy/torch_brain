#!/bin/bash
#SBATCH --job-name=single_gpu_mila
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=48G
#SBATCH --partition=main
#
#set -e
dataset=gillon_richards_responses_2023
export WANDB_PROJECT=openscope_calcium

module load anaconda/3
module load cuda/11.2/nccl/2.8
module load cuda/11.2

#conda activate poyo
source $HOME/poyo_env/bin/activate

# wandb credentials
set -a
source .env
set +a

cd /home/mila/x/xuejing.pan/POYO/project-kirby

# Uncompress the data to SLURM_TMPDIR single node
snakemake --rerun-triggers=mtime -c1 gillon_richards_responses_2023_unfreeze_data
nvidia-smi

srun python /home/mila/x/xuejing.pan/POYO/project-kirby/examples/capoyo/train.py \
        --config-name train_openscope_calcium.yaml data_root=$SLURM_TMPDIR/uncompressed