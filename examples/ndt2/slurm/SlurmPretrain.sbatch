#!/bin/bash
#SBATCH -GH200:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00

cd ..
srun python train.py --config-name pretrain_large