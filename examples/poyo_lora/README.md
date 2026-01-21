# POYO LoRA Finetuning 🧠

Example for finetuning POYO models using Low-Rank Adaptation (LoRA).

LoRA enables efficient finetuning by decomposing weight updates into low-rank matrices,
significantly reducing the number of trainable parameters while maintaining performance.

**Reference:** Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021) [[arXiv]](https://arxiv.org/abs/2106.09685)

---

## Installation

```bash
pip install pytorch_brain lightning==2.3.3 wandb~=0.15
```

---

## Quick Start

### 1. Prepare Data

Download the dataset using [brainsets](https://github.com/neuro-galaxy/brainsets):

```bash
brainsets prepare perich_miller_population_2018
```

### 2. Finetune with LoRA

To finetune a pretrained POYO model using LoRA:

```bash
python train.py ckpt_path="/path/to/pretrained/model.ckpt"
```

For illustration purposes, you can also run without a checkpoint (uses randomly initialized weights):

```bash
python train.py
```

---

## Configuration

### LoRA Parameters

The LoRA configuration is defined in `configs/defaults.yaml` and can be overridden:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora.rank` | 16 | LoRA rank (r) - lower values = fewer parameters |
| `lora.alpha` | 16.0 | LoRA scaling factor (α) - controls adaptation strength |
| `lora.dropout` | 0.0 | Dropout rate for LoRA layers |
| `lora.init_scale` | 0.01 | Initialization scale for LoRA matrix A |
| `lora.target_modules` | ["to_q", "to_kv", "to_qkv", "to_out"] | Module patterns to apply LoRA |
| `lora.target_projections` | ["q", "k", "v", "out"] | Projections within attention to adapt |

### Example: Custom LoRA Configuration

```bash
python train.py \
    ckpt_path="/path/to/model.ckpt" \
    lora.rank=8 \
    lora.alpha=32.0 \
    lora.target_projections="[q,v]"
```

### Target Modules

LoRA can be applied to different attention components:

- `to_q`: Query projection
- `to_k`: Key projection  
- `to_v`: Value projection
- `to_kv`: Combined key-value projection
- `to_qkv`: Combined query-key-value projection
- `to_out`: Output projection

For selective adaptation, you can target specific projections within combined layers:

```yaml
lora:
  target_modules:
    - "to_qkv"
  target_projections:
    - "q"  # Only adapt query projection within to_qkv
    - "v"  # Only adapt value projection within to_qkv
```

---

## How It Works

### LoRA Decomposition

LoRA decomposes weight updates into two low-rank matrices:

```
ΔW = (α/r) × A × B
```

Where:
- `A ∈ R^(d×r)` - Down-projection matrix
- `B ∈ R^(r×k)` - Up-projection matrix  
- `r` - Rank (typically much smaller than d and k)
- `α` - Scaling factor

### Parameter Efficiency

For a linear layer with dimensions `(d_in, d_out)`:
- Full finetuning: `d_in × d_out` parameters
- LoRA: `r × (d_in + d_out)` parameters

With rank `r=16` and `d_in=d_out=512`:
- Full: 262,144 parameters
- LoRA: 16,384 parameters (~6% of full)

### What Gets Trained

During LoRA finetuning, the following are trainable:
1. **LoRA matrices** (A and B) in target modules
2. **Unit embeddings** - for new neurons
3. **Session embeddings** - for new sessions
4. **Readout layer** - task-specific output

The base model weights remain frozen.

---

## Model Configs

Two model configurations are provided:

| Config | Parameters | Description |
|--------|------------|-------------|
| `poyo_1.3M.yaml` | ~1.3M | Smaller model for quick experiments |
| `poyo_11.8M.yaml` | ~11.8M | Larger model matching POYO-MP |

To use a different model:

```bash
python train.py \
    model=poyo_1.3M.yaml \
    ckpt_path="/path/to/model.ckpt"
```

---

## Tips for Finetuning

1. **Start with default rank**: Rank 16 works well for most cases
2. **Adjust alpha**: Higher alpha = stronger adaptation. Try `alpha = rank` as a starting point
3. **Target selection**: For efficiency, start with just `q` and `v` projections
4. **Learning rate**: The default learning rate is usually appropriate for LoRA

---

## Cite

If you use this code, please cite:

```bibtex
@inproceedings{
    azabou2023unified,
    title={A Unified, Scalable Framework for Neural Population Decoding},
    author={Mehdi Azabou and Vinam Arora and Venkataramana Ganesh and Ximeng Mao and Santosh Nachimuthu and Michael Mendelson and Blake Richards and Matthew Perich and Guillaume Lajoie and Eva L. Dyer},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}

@article{hu2021lora,
    title={LoRA: Low-Rank Adaptation of Large Language Models},
    author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
    journal={arXiv preprint arXiv:2106.09685},
    year={2021}
}
```
