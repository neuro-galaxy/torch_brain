# POYO 🧠
Official codebase for POYO published at NeurIPS 2023
[[project page]](https://poyo-brain.github.io/)
[[arxiv]](https://arxiv.org/abs/2310.16046)

### Training
To train POYO-MP you can run:
```bash
python train.py --config-name train_poyo_mp.yaml
```

Checkout `configs/base.yaml` and `configs/train_poyo_mp.yaml` for all configurations
available.

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
