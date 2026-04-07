# Neuroprobe Eval (Public, Neuroprobe2025)

Public release surface for evaluating Neuroprobe2025 with Hydra-based configs.

## Scope

Public code path supports:
- dataset provider: `neuroprobe2025`
- regimes: `SS-SM`, `SS-DM`, `DS-DM`
- models: `logistic`, `mlp`, `cnn`

## Layout

- `run_eval.py`: main entrypoint
- `scripts/`: simple public regime runners
- `models/`: public model implementations
- `preprocessors/`: public preprocessors
- `conf/`: public Hydra configs

## Quick Start

From the examples parent directory:

```bash
cd torch_brain/examples
python -m neuroprobe_eval.run_eval --help
```

### Run by regime (recommended)

```bash
./neuroprobe_eval/scripts/run_eval.sh
```

Edit constants at the top of `scripts/run_eval.sh`:

```bash
REGIME
PATHS_CFG
TEST_SUBJECT
TEST_SESSION
TASK
LABEL_MODE
MODEL
PREPROCESSOR
OUTPUT_GROUP
```

## Environment Overrides for Scripts

Scripts are constant-driven; adjust values directly in the files.

## Direct Hydra Example

```bash
python -m neuroprobe_eval.run_eval \
  paths=<paths_cfg> \
  dataset.provider=neuroprobe2025 \
  dataset.regime=SS-SM \
  dataset.task=onset \
  dataset.test_subject=2 \
  dataset.test_session=0 \
  model=logistic \
  preprocessor=laplacian_stft \
  wandb.enabled=false
```
