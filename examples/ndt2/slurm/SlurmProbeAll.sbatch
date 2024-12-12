#!/bin/bash
#SBATCH -GH200:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00

cd ..
srun python train.py --config-name probe_block
srun python train.py --config-name probe_choice
srun python train.py --config-name probe_wheel
srun python train.py --config-name probe_whisker 

srun python train.py --config-name probe_block wandb.run_name=probe_block_long_finetune_2e-4 optimizer.lr=2e-4
srun python train.py --config-name probe_choice wandb.run_name=probe_choice_long_finetune_2e-4 optimizer.lr=2e-4
srun python train.py --config-name probe_wheel wandb.run_name=probe_wheel_long_finetune_2e-4 optimizer.lr=2e-4
srun python train.py --config-name probe_whisker wandb.run_name=probe_whisker_long_finetune_2e-4 optimizer.lr=2e-4

srun python train.py --config-name probe_block wandb.run_name=probe_block_long_finetune_5e-5 optimizer.lr=5e-5
srun python train.py --config-name probe_choice wandb.run_name=probe_choice_long_finetune_5e-5 optimizer.lr=5e-5
srun python train.py --config-name probe_wheel wandb.run_name=probe_wheel_long_finetune_5e-5 optimizer.lr=5e-5
srun python train.py --config-name probe_whisker wandb.run_name=probe_whisker_long_finetune_5e-5 optimizer.lr=5e-5

srun python train.py --config-name probe_block wandb.run_name=probe_block_long_finetune_4e-4 optimizer.lr=4e-4
srun python train.py --config-name probe_choice wandb.run_name=probe_choice_long_finetune_4e-4 optimizer.lr=4e-4
srun python train.py --config-name probe_wheel wandb.run_name=probe_wheel_long_finetune_4e-4 optimizer.lr=4e-4
srun python train.py --config-name probe_whisker wandb.run_name=probe_whisker_long_finetune_4e-4 optimizer.lr=4e-4