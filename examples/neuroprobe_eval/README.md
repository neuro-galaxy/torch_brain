# Modular Neuroprobe Evaluation

A simplified, modular implementation for evaluating models on Neuroprobe, designed to replicate `neuroprobe_linear_baselines.ipynb` and easily extend to PyTorch models.

## Structure


- `run_eval.py`: Main entry point (like `run_train.py`)
- `sklearn_runner.py`: Runner for sklearn models
- `torch_runner.py`: Runner for PyTorch models
- `models/`: Model implementations (one file per model)
- `preprocessors/`: Preprocessor implementations (one file per preprocessor)
- `utils/`: Data loading and logging utilities
- `conf/`: Hydra configuration files
- `aggregate_results.py`: Aggregate and export results
- `run_all.sh`: Batch script to run all experiments

## Quick Start

### Replicate Notebook Workflow

Run all experiments (tasks × subjects × trials × folds) for all preprocessors:

```bash
./run_all.sh /path/to/processed/data
```

This will:
1. Run all experiments for raw, stft, and laplacian-stft preprocessors
2. Aggregate results into leaderboard format
3. Export to DataFrame CSV with summary statistics

### Single Experiment

Run a single experiment:

```bash
python run_eval.py \
    data_source=processed \
    processed_data_path=/path/to/processed/data \
    subject_id=1 \
    trial_id=1 \
    eval_name=onset \
    model=logistic \
    preprocessor=raw
```

## Data Sources

**Processed data (recommended for notebook replication):**
- Uses pre-computed splits (faster, simpler)
- No time bins, just folds
- Requires processed H5 files

```bash
python run_eval.py \
    data_source=processed \
    processed_data_path=/path/to/processed/data \
    subject_id=1 trial_id=1 eval_name=onset
```

**Raw data (more flexible):**
- Creates splits on-the-fly
- Supports time bins and different split types
- Requires raw H5 files

```bash
python run_eval.py \
    data_source=raw \
    subject_id=1 trial_id=1 eval_name=onset \
    splits_type=WithinSession
```

## Models

**Logistic Regression (sklearn):**
```bash
python run_eval.py model=logistic ...
```

**MLP (PyTorch):**
```bash
python run_eval.py model=mlp ...
```

**CNN (PyTorch):**
```bash
python run_eval.py model=cnn ...
```

**Transformer (PyTorch):**
```bash
python run_eval.py model=transformer ...
```

## Preprocessors

**Raw:**
```bash
python run_eval.py preprocessor=raw ...
```

**STFT:**
```bash
python run_eval.py preprocessor=stft ...
```

**Laplacian + STFT:**
```bash
python run_eval.py preprocessor=laplacian-stft ...
```

## Output

Results are saved as JSON files compatible with Neuroprobe leaderboard format:

```
eval_results/
  Within-Session/
    logistic_voltage/
      population_btbank1_1_onset.json
      ...
```

### Aggregating Results

**Leaderboard format (per-task files):**
```bash
python aggregate_results.py \
    --split-type Within-Session \
    --task all
```

**DataFrame format (notebook-style):**
```bash
python aggregate_results.py \
    --to-dataframe \
    --output-csv results.csv \
    --print-summary
```

## Adding New Models

1. Create `models/my_model.py`:
```python
from .base_model import BaseModel
from . import register_model

@register_model("my_model")
class MyModel(BaseModel):
    def __init__(self, cfg):
        super().__init__()
        # Initialize model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Train model
        pass
    
    def predict_proba(self, X):
        # Return probabilities
        pass
```

2. Create `conf/model/my_model.yaml`:
```yaml
name: my_model
param1: value1
```

3. Use it:
```bash
python run_eval.py model=my_model ...
```

## Adding New Preprocessors

1. Create `preprocessors/my_preprocessor.py`:
```python
from .base_preprocessor import BasePreprocessor
from . import register_preprocessor

@register_preprocessor("my_preprocessor")
class MyPreprocessor(BasePreprocessor):
    def preprocess(self, data, electrode_labels):
        # Apply preprocessing
        return processed_data
```

2. Create `conf/preprocessor/my_preprocessor.yaml`:
```yaml
name: my_preprocessor
param1: value1
```

3. Use it:
```bash
python run_eval.py preprocessor=my_preprocessor ...
```

## Configuration

All configuration via Hydra YAML files. Main config: `conf/config.yaml`

Key options:
- `data_source`: "raw" or "processed"
- `subject_id`, `trial_id`, `eval_name`: Evaluation parameters
- `model`: Model name (logistic, mlp, cnn, transformer)
- `preprocessor`: Preprocessor name (raw, stft, laplacian-stft)
- `processed_data_path`: Path to processed data (for processed source)
- `seed`: Random seed (default: 42)
- `save_dir`: Output directory (default: "eval_results")
- `overwrite`: Overwrite existing results (default: false)
- `verbose`: Print verbose logs (default: true)

## Design Principles

1. **Runner Pattern**: Separate runners for sklearn vs PyTorch (like PopulationTransformer)
2. **Simple Entry Point**: `run_eval.py` just orchestrates: config → data → model → runner → save
3. **Config-Driven**: Everything configured via Hydra YAML files
4. **Registry Pattern**: Models and preprocessors auto-discovered via decorators
5. **Modularity**: One file per model/preprocessor for easy maintenance
