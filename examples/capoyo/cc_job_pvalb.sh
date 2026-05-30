#!/bin/bash
#SBATCH --account=rrg-tyrell-ab
#SBATCH --job-name=capoyo_allen.txt
#SBATCH --output=slurm_output-%j.txt
#SBATCH --error=slurm_error-%j.txt
#SBATCH --ntasks-per-node=4
#SBATCH --mem=256GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=12:0:0

module load StdEnv/2020 python/3.9 #httpproxy
cd /home/$USER
source ENV/bin/activate
#wandb offline

cd project-kirby
snakemake --rerun-triggers=mtime --cores 4 allen_brain_observatory_calcium_unfreeze

# Important info for parallel GPU processing
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export NCCL_BLOCKING_WAIT=1

cd examples/capoyo
export WANDB_mode=offline

srun python train.py --config-name train_model_3_visp.yaml log_dir=/home/mazabou/scratch \
	dataset=allen_brain_observatory_calcium_sub_pvalb.yaml \
	name=capoyo_allen_pvalb \
	eval_epochs=1 \
	epochs=2000 \
	data_root=${SLURM_TMPDIR}/uncompressed \
	pct_start=0.9 \
	batch_size=128 \
	base_lr=1.56e-5 \
	num_workers=4 \
	nodes=1 \
	gpus=4 \
	+wandb_log_model=True
