#!/bin/bash
source ~/.bashrc
module load python/3.9.6
conda activate poyo

# Not thread-safe!
snakemake --nolock --rerun-triggers=mtime --config tmp_dir="$SLURM_TMPDIR" -c1 "$1"
