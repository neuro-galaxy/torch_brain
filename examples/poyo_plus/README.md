# POYO+ 🧠

Official codebase for POYO+ from ICLR 2025 ([Paper Link](https://proceedings.iclr.cc/paper_files/paper/2025/file/953390c834451505703c9da45de634d8-Paper-Conference.pdf)).

---

POYO+ is a multi-task version of POYO.
This is an example training script for the Calcium POYO+ model in [Azabou and Pan et al. 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/953390c834451505703c9da45de634d8-Paper-Conference.pdf), corresponding to the module `torch_brain.models.CalciumPOYOPlus`.


## Training POYO+ on Calcium Traces

**Installing necessary packages**
```bash
pip install torch_brain "lightning>=2.6.0" wandb brainsets
```

> [!Note]
> `lightning>=2.6.0` is required because `train.py` passes the
> `weights_only=False` argument to `Trainer.fit`, which was added in 2.6.0.

**Data Preparation**
There are 1304 sessions in the full Calcium POYO+ model and 30 holdout drifting gratings sessions.
The raw data for all sessions is ~360GB and processed data uses ~58GB.
To prepare the data, run:

```bash
brainsets prepare allen_visual_coding_ophys_2016
```

### Generating the full multi-task dataset config

The full multi-task dataset config groups every Allen Visual Coding session by
the exact set of tasks it supports. Because that file is ~1.7k lines, it is
generated rather than checked in. To produce it locally:

```bash
cd examples/poyo_plus/scripts
python generate_config.py
```

This reads `task_contents.csv` (which lists, per session, the tasks available)
and writes the full grouped dataset file to
`examples/poyo_plus/configs/dataset/calcium_poyo_plus.yaml`. A small reference
stub lives at `examples/poyo_plus/configs/dataset/calcium_poyo_plus_example.yaml`
so you can see the schema without running the generator.

**Training**

```bash
python train.py --config-name=train_calcium_poyo_plus.yaml
```

Check out `configs/train_calcium_poyo_plus.yaml` for full-model config and `configs/train_calcium_poyo_plus_single_session.yaml` for a single-session example.

## Pretrained weights

A Calcium POYO+ checkpoint trained on the full 1304-session Allen Visual
Coding corpus is available here:

- [Calcium POYO+ checkpoint (epoch 414, ~275 MB)](https://drive.google.com/file/d/1zKzR9YU1dDfqe9ygcBLfdj8Masjk_I-l/view?usp=sharing)

Download it and load it with the same model config used at training time
(`configs/model/calcium_poyo_plus.yaml`):

The `unit_emb` and `session_emb` `InfiniteVocabEmbedding` layers will be
materialized automatically from the vocabulary saved alongside the weights
(116702 units, 1306 sessions).

### Test results

Held-out test metrics for this checkpoint, averaged across recordings (overall
`average_test_metric = 0.3352`):

| Task                                 | Metric                            |  Value |
| ------------------------------------ | --------------------------------- | -----: |
| drifting_gratings_orientation        | Accuracy                          | 0.4719 |
| drifting_gratings_temporal_frequency | Accuracy                          | 0.4503 |
| static_gratings_orientation          | Accuracy                          | 0.3347 |
| static_gratings_phase                | Accuracy                          | 0.2664 |
| static_gratings_spatial_frequency    | Accuracy                          | 0.4455 |
| natural_scenes                       | Accuracy                          | 0.0968 |
| natural_movie_one_frame              | mean(Accuracy, WithinDeltaAcc)    | 0.4296 |
| natural_movie_two_frame              | mean(Accuracy, WithinDeltaAcc)    | 0.3685 |
| natural_movie_three_frame            | mean(Accuracy, WithinDeltaAcc)    | 0.1598 |
| pupil_location                       | MeanSquaredError (z-normalized)   | 0.2791 |
| running_speed                        | MeanSquaredError (z-normalized)   | 0.4021 |

> [!Note]
> Performnce numbers differ from those reported in the paper. In particular, our
> `running_speed` prediction target was actually the monotonically-increasing
> timestamps rather than the m/s running-speed value, so that row is not
> directly comparable. The remaining differences are attributable to small
> hyperparameter/implementation deltas and stochasticity in training.

## Finetuning

To finetune a pre-trained model, download the checkpoint above and run:

```bash
python finetune.py ckpt_path="/path/to/epoch_epoch=414.ckpt"
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

