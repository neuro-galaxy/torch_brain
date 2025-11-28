# POYO 🧠
Official codebase for POYO published at NeurIPS 2023
[[project page]](https://poyo-brain.github.io/)
[[arxiv]](https://arxiv.org/abs/2310.16046)

### Installation

```bash
pip install pytorch_brain lightning==2.3.3 wandb~=0.15
```

### Training POYO-MP
To train POYO-MP you first need to download the [`perich_miller_population_2018`](https://brainsets.readthedocs.io/en/latest/glossary/brainsets.html#perich-miller-population-2018) data using [brainsets](https://github.com/neuro-galaxy/brainsets).

```bash
brainsets prepare perich_miller_population_2018
```

Then you can train POYO-MP by running:

```bash
python train.py --config-name train_poyo_mp.yaml
```

Checkout `configs/defaults.yaml` and `configs/train_poyo_mp.yaml` for all configurations
available.

### Training POYO-1
To train POYO-1 you first need to download all datasets using `brainsets`.

```bash
brainsets prepare perich_miller_population_2018
brainsets prepare churchland_shenoy_neural_2012
brainsets prepare flint_slutzky_accurate_2012
brainsets prepare odoherty_sabes_nonhuman_2017
```

Then you can train POYO-1 by running:

```bash
python train.py --config-name train_poyo_1.yaml
```

Checkout `configs/base.yaml` and `configs/train_poyo_1.yaml` for all configurations available.
>>>>>>> origin/main

### Finetuning
To finetune POYO-MP you can run:
```bash
python train.py --config-name train_poyo_mp.yaml \
    ckpt_path=<path-to-checkpoint> \
    finetuning.enable=true \
    finetuning.freeze_perceiver_until_epoch=10  # set to -1 to do unit-identification
```


## Cite
Please cite [our paper](https://papers.nips.cc/paper_files/paper/2023/hash/8ca113d122584f12a6727341aaf58887-Abstract-Conference.html) if you use this code in your own work:

```bibtex
@inproceedings{
    azabou2023unified,
    title={A Unified, Scalable Framework for Neural Population Decoding},
    author={Mehdi Azabou and Vinam Arora and Venkataramana Ganesh and Ximeng Mao and Santosh Nachimuthu and Michael Mendelson and Blake Richards and Matthew Perich and Guillaume Lajoie and Eva L. Dyer},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```
