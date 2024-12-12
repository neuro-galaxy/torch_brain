#!/bin/bash
#SBATCH -J IBL-NDT2
#SBATCH -o /home/hice1/aandre8/scratch/alex/logs/SlurmSweep.out
#SBATCH -e /home/hice1/aandre8/scratch/alex/logs/SlurmSweep.err
#SBATCH --cpus-per-gpu=16
#SBATCH -G H100:1
#SBATCH -t 12:00:00
#SBATCH --mem=64G

source /home/hice1/aandre8/scratch/alex/venv/bin/activate
srun wandb agent --count 1 aandre8-gatech/NDT2_IBL/1attojpf
