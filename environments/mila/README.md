# Running on the mila cluster

Mila is a SLURM-based cluster. `run.sh` gives an example one-node pipeline, `run.sh` 
`run_parallel.sh` gives an example single-node, multi-GPU pipeline, `run_multi_node.sh`, 
a multi-node multi-GPU pipeline. In practice, multi-node, multi-GPU runs are only 
feasible during deadtimes during conferences, but still, it works!

## Datasets

We keep a canonical version of the raw data in the common location at 
`/network/projects/neuro-galaxy/data/raw`. This prevents having to download the data 
multiple times, which could easily take a day. Processed, compressed data is stored in
your personal scratch folder, which prevents undefined behaviour when multiple people 
are modifying the same `process_data.py` pipeline.

Because mila is SLURM-based, data is first copied to the local node (SLURM_TMPDIR), 
then processed in jobs. Because the file system is distributed and doesn't like to deal 
with small files, we use tarballs compressed with lz4, which is a ridiculously fast 
compression algorithm. Typically, the data will be processed in four stages:

* download to `/network/projects/neuro-galaxy/data/raw`
* processed to `$SLURM_TMPDIR/compressed`
* frozen (i.e. compressed) to `~/scratch/data/compressed`
* unfrozen (i.e. decompressed) to `$SLURM_TMPDIR/uncompressed`

When files are processed by job and subsequently frozen, once the job is done, the files
in $SLURM_TMPDIR are deleted. This is an unavoidable consequence of the data processing 
DAG and the constraints of SLURM. Thus, if we call, e.g. `willett_shenoy_unfreeze`, it 
will first attempt to re-process the data, complaining that the intermediate files don't 
exist, e.g.:

```
job                            count
---------------------------  -------
willett_shenoy_freeze              1
willett_shenoy_prepare_data        1
willett_shenoy_unfreeze            1
total                              3

Select jobs to execute...

[Thu Nov 16 10:06:57 2023]
rule willett_shenoy_prepare_data:
    input: /network/projects/neuro-galaxy/data/raw/willett_shenoy/handwritingBCIData/Datasets/t5.2019.11.25/singleLetters.mat, data/scripts/willett_shenoy/prepare_data.py
    output: /Tmp/slurm.3839591.0/processed/willett_shenoy/description.mpk
    jobid: 2
    reason: Missing output files: /Tmp/slurm.3839591.0/processed/willett_shenoy/description.mpk
```

A workaround is to use timestamps, and not the presence of intermediate files as 
the trigger for parent rules, e.g.:

```
snakemake --rerun-triggers=mtime -c1 willett_shenoy_unfreeze
```

This will not trigger the creation of intermediate artifacts provided the timestamps of 
the artifacts make sense.

## Environment

Set up a conda environment with the right packages, as defined in `requirements.txt`.

## Setting up a compute environment
When building the dataset on the Mila clusters, it's important to have a sufficient 
amount of memory and compute or else errors will occur, resulting in incomplete 
datasets. Here is an example allocation request that works:
```
salloc -c 10 --mem 32G --gres gpu:1
```

## Partitions

Mila has a number of partitions, `unkillable`, `short-unkillable`, `main` and `long`. 
Use 1-GPU `unkillable` jobs for debugging. Run a 4-GPU job on `short-unkillable` to get 
very quick results (3 hours max, but equivalent to 12 hours of a 1 GPU job). Use the 
main and long partitions for longer jobs.

[Reference](https://docs.mila.quebec/Userguide.html#partitioning)


## wandb credentials

Store them in `.env` in the root of the project. This file is ignored by git. It should
look like:

```
WANDB_PROJECT=poyo
WANDB_ENTITY=neuro-galaxy
WANDB_API_KEY=<secret-key>
```

Get the API key from the wandb website.

## mila cluster $SLURM_TMPDIR issue
(updated 04/25/2024 - Krystal)

When running an interactive session on the mila cluster (e.g. `mila_code` in VScode or `salloc` jobs), it is possible that $SLURM_TMPDIR is not expanded and cause a Snakemake error:
```
/bin/bash: line 1: SLURM_TMPDIR: unbound variable
```

Appending the following code to the `~/.bashrc` file to specify the temporary folder can solve this issue.
```
export SLURM_TMPDIR="/tmp"
```

After modifying, you only need to run it once:
```
source ~/.bashrc
```

Now $SLURM_TMPDIR will expand and the unbound variable issue is solved.
