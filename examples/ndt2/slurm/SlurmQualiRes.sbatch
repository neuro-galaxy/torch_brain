#!/bin/bash
#SBATCH -GA100:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=64G
#SBATCH --time=00:10:00

cd ..
eid='db4df448-e449-4a6f-a0e7-288711e7a75a'
srun python eval_ndt2_single_task.py --eid ${eid} --behavior wheel --ckpt_postfix '_long_finetune_5e-5'
srun python eval_ndt2_single_task.py --eid ${eid} --behavior whisker --ckpt_postfix '_long_finetune_4e-4'
