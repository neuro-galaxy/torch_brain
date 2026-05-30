# [WIP] CaPOYO: POYO applied to Calcium Imaging Data

### Datasets
To download and prepare the openscope calcium dataset, run the following inside the 
`project-kirby` directory:
```bash
snakemake --cores 8 gillon_richards_responses_2023
```
This dataset extracts the Allen Brain Observatory calcium traces from 433 sessions, including only the `drifting grating` stimuli. 
To download and prepare data, Run:
```bash
snakemake --cores 8 allen_brain_observatory_calcium_unfreeze
```

### Training CaPOYO
(remember to overwrite data root to be your processed data root)
```bash
python train.py --config-name train_openscope_calcium.yaml data_root=/kirby/processed
```
```bash
python train.py --config-name train_allen_bo.yaml data_root=/kirby/processed
```