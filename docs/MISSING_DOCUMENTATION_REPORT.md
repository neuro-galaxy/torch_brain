# Missing Documentation Report

This report outlines the documentation gaps found in the torch_brain package. The issues are categorized by severity and type.

---

## 1. Empty or Placeholder Documentation Files

### High Priority

| File | Issue |
|------|-------|
| `docs/source/concepts/getting_started.rst` | File is empty - should contain an introductory guide |
| `docs/source/tutorials/README.rst` | Contains only a placeholder, no actual tutorials |

---

## 2. Missing Module/Class Documentation in RST Files

### `torch_brain.utils` (`docs/source/package/utils.rst`)

The utils.rst file only documents tokenizer functions. The following are missing:

| Module/Function | Description |
|-----------------|-------------|
| `seed_everything` | Function to seed all random number generators |
| `get_sinusoidal_encoding` | Function for sinusoidal position encoding |
| `resolve_weights_based_on_interval_membership` | Utility to compute weights based on interval membership |
| `isin_interval` | Check if timestamps fall within intervals |
| `prepare_for_readout` | Prepare data for single-task readout |

### `torch_brain.utils.stitcher` (Not documented at all)

The stitcher module contains important evaluation utilities that are referenced in the multitask_readout concept guide but have no API reference:

| Class/Function | Description |
|----------------|-------------|
| `stitch` | Pool predictions by timestamp (mean for continuous, mode for categorical) |
| `DecodingStitchEvaluator` | Lightning callback for single-task evaluation with stitching |
| `MultiTaskDecodingStitchEvaluator` | Lightning callback for multi-task evaluation with stitching |
| `DataForDecodingStitchEvaluator` | Dataclass for passing data to evaluator |
| `DataForMultiTaskDecodingStitchEvaluator` | Dataclass for passing multi-task data to evaluator |

### `torch_brain.utils.callbacks`, `binning`, `gradient_rescale`

These modules exist but are not documented or exposed in the public API. Consider either:
- Adding documentation if they are meant to be public
- Removing from the package if internal-only

---

## 3. Missing Docstrings in Code

### `torch_brain/nn/feedforward.py`

| Class | Issue |
|-------|-------|
| `GEGLU` | Missing Args section in docstring |

The forward method should document input/output shapes.

### `torch_brain/models/capoyo.py`

| Class | Issue |
|-------|-------|
| `CaPOYO` | Class docstring is minimal - missing detailed Args documentation similar to POYOPlus |

### `torch_brain/models/poyo.py`

| Function | Issue |
|----------|-------|
| `poyo_mp` | Factory function has no docstring explaining its purpose or return value |

### `torch_brain/utils/` files

| File | Issue |
|------|-------|
| `seed_everything.py` | Missing module-level docstring and function docstring |
| `sinusoidal_encoding.py` | Missing docstrings |
| `binning.py` | Missing docstrings |
| `callbacks.py` | Missing docstrings |
| `gradient_rescale.py` | Missing docstrings |

---

## 4. Inconsistencies in Documentation

### Split naming convention

In `docs/source/concepts/sampler.rst`:
- Line 272 uses `"val"` for validation split
- Line 84 uses `"valid"` for validation split

**Recommendation:** Standardize on one naming convention throughout the documentation.

---

## 5. Missing Conceptual Documentation

### Suggested New Concept Guides

1. **Getting Started Guide** (`getting_started.rst`)
   - Installation verification
   - Quick example of loading data and running a model
   - Overview of the package structure

2. **Data Pipeline Guide**
   - How to prepare data using brainsets
   - How the Dataset, Sampler, and Collate work together
   - Transform pipeline explanation

3. **Model Training Guide**
   - Setting up a training loop with Lightning
   - Using the evaluation callbacks
   - Checkpointing and model loading

4. **Custom Modality Registration**
   - When and how to register custom modalities
   - Examples beyond the built-in modalities

---

## 6. Documentation Format Issues

### Docstring Style Consistency

Some files use Google-style docstrings while others use NumPy-style. Recommend standardizing on one style (Google style appears more common in the codebase).

### Type Hints in Docstrings

Many functions have type hints in function signatures but don't repeat them in docstrings, which is good. However, some older docstrings do repeat types (e.g., `(int):`) - these could be simplified.

---

## Summary

| Category | Count |
|----------|-------|
| Empty/placeholder files | 2 |
| Missing RST documentation | ~15 classes/functions |
| Missing code docstrings | ~10 classes/functions |
| Inconsistencies | 1 |
| Missing concept guides | 4 |

### Recommended Priority

1. **High:** Fill in `getting_started.rst` - this is the entry point for new users
2. **High:** Document the stitcher module - referenced heavily in existing docs
3. **Medium:** Add utils module documentation
4. **Medium:** Add docstrings to undocumented classes
5. **Low:** Standardize docstring format and split naming

