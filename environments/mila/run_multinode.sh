#!/bin/bash
#SBATCH --job-name=multi-node-mila
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --mem=496GB
#SBATCH --switch=1
#SBATCH --partition=long

dataset=perich_single_session
rule=perich_miller_unfreeze

module load anaconda/3
module load cuda/11.2/nccl/2.8
module load cuda/11.2

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
snakemake --rerun-triggers=mtime --config tmp_dir="$SLURM_TMPDIR" -c1 --until $rule
srun copy_data.sh $rule

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
    eval_epochs=1  \
    epochs=1 \
    pct_start=0.9 \
    batch_size=128 \
    name=multi_node_mila \
    base_lr=1e-5 \
    precision=16 \
    num_workers=6 \
    model=poyo_single_session \
    nodes=2 \
    gpus=4