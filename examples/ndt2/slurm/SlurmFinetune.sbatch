#!/bin/bash
#SBATCH -GH100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

cd ..
srun python train.py --config-name finetune