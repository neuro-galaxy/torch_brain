#!/bin/bash
#SBATCH --job-name=multi-run
#SBATCH --output=slurm_output.txt
#SBATCH --error=slurm_error.txt
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24GB
#SBATCH --time=1-23:59:59
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=main
#SBATCH --gres=gpu:1

# For training, one can also use the following options:
#E.g. SBATCH --partition=unkillable and SBATCH --gres=gpu:a100

dataset=odoherty_single_session_lfp

module load anaconda/3
module load cuda/12.0

conda activate poyo

# wandb credentials
set -a
source .env
set +a

# Unpack data to $SLURM_TMPDIR, which also symlinked via /home/mila/p/patrick.mineault/slurm_tmpdir
snakemake --rerun-triggers=mtime --config tmp_dir=$SLURM_TMPDIR -c1 odoherty_sabes_unfreeze
# snakemake --rerun-triggers=mtime --config tmp_dir=$SLURM_TMPDIR -c1 willett_shenoy_unfreeze
# snakemake --rerun-triggers=mtime --config tmp_dir=$SLURM_TMPDIR -c1 perich_miller_unfreeze

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export NCCL_SOCKET_IFNAME=^docker0,lo

echo $MASTER_ADDR:$MASTER_PORT

if [ ${ckpt} = true ] ; then
    name="${dataset}_continue"
    unique_args=(
        "name=${name}"
        "finetune_path=logs/lightning_logs/tvh4iucy/checkpoints/last.ckpt"
        "finetune_epochs=40"
    )
else
    name="${dataset}_scratch"
    unique_args=("name=${name}")
fi

# Run experiments
pwd
which python
python scripts/train_single_session_lightning.py \
    data_root=$SLURM_TMPDIR/uncompressed/ \
    train_datasets=$dataset \
    val_datasets=$dataset \
    eval_epochs=10  \
    epochs=500 \
    pct_start=0.9 \
    batch_size=64 \
    name=${dataset} \
    base_lr=1e-5 \
    precision=16 \
    num_workers=4 \
    "${unique_args[@]}"
