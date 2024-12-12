#!/bin/bash
#SBATCH -GA100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00

cd ..
srun python train.py --config-name probe_choice
