#!/bin/bash
#SBATCH --job-name=capoyo-full
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --mem-per-gpu=128G
#SBATCH --cpus-per-gpu=16
#SBATCH --output=logs/capoyo-full-%j.out
#SBATCH --error=logs/capoyo-full-%j.err
source /vast/projects/dyer1/lab/user/ian/envs/torch_brain_env/bin/activate
cd ~/torch_brain
srun python examples/poyo_plus/train.py --config-name=train_capoyo.yaml \
    model=capoyo.yaml \
    dataset=capoyo.yaml \
    wandb.run_name="capoyo-full" \
    ckpt_path="logs/poyo/bwash9hq/checkpoints/last.ckpt"
