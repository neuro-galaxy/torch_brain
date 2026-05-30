#!/bin/bash
#SBATCH --job-name=multi-run
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=124GB
#SBATCH --time=1:00:00
#SBATCH --switch=1

set -e
# For training, one can also use the following options:
#E.g. SBATCH --partition=unkillable and SBATCH --gres=gpu:a100

dataset=perich_multi_session

source ~/.bashrc
module load python/3.9.6
module load cuda/11.2
module load httpproxy
conda activate poyo

# wandb credentials
set -a
source .env
set +a

# Unpack data to $SLURM_TMPDIR. This needs to be done once per node. 
# However, there is a slim chance the prepared data is stale, and if multiple nodes 
# try to reconstruct the data and to write to the same place to disk at the same time
# we'd have a bad time. Hence we run snakemake on the master node --until the unfreeze 
# rule, then run the thread-safe unfreeze rule proper on each node. Note that 
# srun fires of one process per node.
rule=perich_miller_unfreeze
snakemake --rerun-triggers=mtime --config tmp_dir="$SLURM_TMPDIR" -c1 --until $rule
srun copy_data.sh $rule

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export NCCL_BLOCKING_WAIT=1

echo $MASTER_ADDR:$MASTER_PORT

nvidia-smi

# Run experiments
pwd
which python
srun python train.py \
    data_root=$SLURM_TMPDIR/uncompressed/ \
    train_datasets=$dataset \
    val_datasets=$dataset \
    eval_epochs=2  \
    epochs=1 \
    pct_start=0.9 \
    batch_size=96 \
    name=benchmark_two_node_narval \
    base_lr=1e-5 \
    precision=16 \
    num_workers=6 \
    model=poyo_1 \
    nodes=2 \
    gpus=1