# POYO Example

A simple example training script for [POYO](https://poyo-brain.github.io/)

### Setup

In addition to installing `torch_brain`, you need to:
```bash
pip install wandb brainsets
```

### Training POYO-MP
To train POYO-MP you first need to download and preprocess the [`perich_miller_population_2018`](https://brainsets.readthedocs.io/en/latest/_generated/brainsets.datasets.PerichMillerPopulation2018.html#brainsets.datasets.PerichMillerPopulation2018) data using [brainsets](https://github.com/neuro-galaxy/brainsets).

```bash
brainsets config --raw-dir data/raw --processed-dir data/processed
brainsets prepare perich_miller_population_2018
```

Then you can train POYO-MP by running:

```bash
python train_simple.py --config-name train_poyo_mp data_root=data/processed
```

Checkout `configs/defaults.yaml` and `configs/train_poyo_mp.yaml` for all configurations available.

### Training POYO-1
To train POYO-1 you first need to download all datasets using `brainsets`.

```bash
brainsets config --raw-dir data/raw --processed-dir data/processed
brainsets prepare perich_miller_population_2018
brainsets prepare churchland_shenoy_neural_2012
brainsets prepare flint_slutzky_accurate_2012
brainsets prepare odoherty_sabes_nonhuman_2017
```

Then you can train POYO-1 by running:

```bash
python train_simple.py --config-name train_poyo_1 data_root=data/processed
```

Checkout `configs/defaults.yaml` and `configs/train_poyo_1.yaml` for all configurations available.

## Citation for POYO:

```bibtex
@inproceedings{
    azabou2023unified,
    title={A Unified, Scalable Framework for Neural Population Decoding},
    author={Mehdi Azabou and Vinam Arora and Venkataramana Ganesh and Ximeng Mao and Santosh Nachimuthu and Michael Mendelson and Blake Richards and Matthew Perich and Guillaume Lajoie and Eva L. Dyer},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```
