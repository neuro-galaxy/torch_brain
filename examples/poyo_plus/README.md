# POYO+ 🧠

Official codebase for POYO+ from [ICLR 2025].  
[[Paper Link]](https://proceedings.iclr.cc/paper_files/paper/2025/file/953390c834451505703c9da45de634d8-Paper-Conference.pdf)

---

POYO+ is a multi-task version of POYO.
This is an example training script for the model in [Azabou and Pan et al. 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/953390c834451505703c9da45de634d8-Paper-Conference.pdf), corresonding to the module `torch_brain.models.CaPOYO`.


## Training POYO+ on Calcium Traces

**Data Preparation**  
There are 1304 sessions in the full CaPOYO model and 30 holdout drifting gratings sessions.
The raw data for all sessions is ~360GB and processed data uses ~58GB.
To prepare the data, run:

```bash
brainsets prepare allen_visual_coding_ophys_2016
```

**Training**  

```bash
python train.py --config-name=train_capoyo.yaml
```

Check out `configs/train_capoyo.yaml` for full-model config and `configs/train_capoyo_single_session.yaml` for a single-session example.


## Finetuning

To finetune a pre-trained model, run:

```bash
python finetune.py ckpt_path="*your_path_here*"
```

- Set which model and dataset to use in `configs/finetune.yaml`.
- Update `ckpt_path` to your model checkpoint location.

---

## Cite

If you use this code, please cite our paper:

```bibtex
@inproceedings{azabou2025multisession,
  author = {Azabou, Mehdi and Pan, Krystal and Arora, Vinam and Knight, Ian and Dyer, Eva and Richards, Blake A},
  booktitle = {International Conference on Learning Representations},
  editor = {Y. Yue and A. Garg and N. Peng and F. Sha and R. Yu},
  pages = {59654--59677},
  title = {Multi-session, multi-task neural decoding from distinct cell-types and brain regions},
  url = {https://proceedings.iclr.cc/paper_files/paper/2025/file/953390c834451505703c9da45de634d8-Paper-Conference.pdf},
  volume = {2025},
  year = {2025}
}
```

