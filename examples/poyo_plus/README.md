# POYO 🧠
Official codebase for POYO+ published at ICLR2025
[[Paper]](https://proceedings.iclr.cc/paper_files/paper/2025/file/953390c834451505703c9da45de634d8-Paper-Conference.pdf)

### Training Calcium POYO

There are 1350 sessions used in the full CaPOYO model.
```bash
brainsets prepare allen_visual_coding_ophys_2016”
```
The raw data for these sessions uses ~381 Gb and the processed data uses ~61 Gb.

To train Calcium POYO (CaPOYO) you can run:
```bash
python train.py --config-name train_capoyo_base.yaml

Checkout `configs/train_capoyo.yaml` for the full model and `configs/train_capoyo_single_session.yaml` for a single session example.

We also include POYO Plus, a multi-task implementation of POYO-1.

### Finetuning

To start finetuning, you can run:
```bash
python finetune.py ckpt_pth="*your_path_here*”
```
You can set the model to be finetune and the dataset to be finetuned on in the finetune.yaml config. 

## Cite
Please cite [our paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/953390c834451505703c9da45de634d8-Paper-Conference.pdf) if you use this code in your own work:

```bibtex
@inproceedings{ICLR2025_953390c8,
 author = {Azabou, Mehdi and Pan, Krystal and Arora, Vinam and Knight, Ian and Dyer, Eva and Richards, Blake A},
 booktitle = {International Conference on Representation Learning},
 editor = {Y. Yue and A. Garg and N. Peng and F. Sha and R. Yu},
 pages = {59654--59677},
 title = {Multi-session, multi-task neural decoding from distinct cell-types and brain regions},
 url = {https://proceedings.iclr.cc/paper_files/paper/2025/file/953390c834451505703c9da45de634d8-Paper-Conference.pdf},
 volume = {2025},
 year = {2025}
}

```

