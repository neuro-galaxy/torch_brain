#!/bin/bash
conda activate poyo
module load python/3.9.6

# Not thread-safe!
snakemake --nolock --rerun-triggers=mtime --config tmp_dir="$SLURM_TMPDIR" -c1 "$1"