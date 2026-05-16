# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added `MultiChannelDatasetMixin` to provide `get_channel_ids` and prefixing interface for EEG-like datasets ([#173](https://github.com/neuro-galaxy/torch_brain/pull/173))
- Added `BinSpikes` transform ([#170](https://github.com/neuro-galaxy/torch_brain/pull/170))
- Added public weights for `CalciumPOYOPlus` ([#198](https://github.com/neuro-galaxy/torch_brain/pull/198))
- Added per-task average metric logging in `MultiTaskDecodingStitchEvaluator` ([#198](https://github.com/neuro-galaxy/torch_brain/pull/198))

### Removed
- Removed `torch_brain.nn.FeedForwad` (too inflexible) ([#204](https://github.com/neuro-galaxy/torch_brain/pull/204))
- Removed `torch_brain.optim` / `SparseLamb` from the public package API ([#203](https://github.com/neuro-galaxy/torch_brain/pull/203))
- Remove `utils/gradient_rescale.py` (legacy) ([#201](https://github.com/neuro-galaxy/torch_brain/pull/201))
- Removed `utils.get_sinusoidal_encoding` (legacy) ([#200](https://github.com/neuro-galaxy/torch_brain/pull/200))

### Changed
- Changed minimum `temporaldata` version to `v0.1.4` ([#209](https://github.com/neuro-galaxy/torch_brain/pull/209))
- Moved `DecodingStitchEvaluator` and `MultiTaskDecodingStitchEvaluator` to `utils/callbacks.py` ([#183](https://github.com/neuro-galaxy/torch_brain/pull/183))
- Replaced `torch_brain.data.dataset.DatasetIndex` with `torch_brain.dataset.DatasetIndex` ([#186](https://github.com/neuro-galaxy/torch_brain/pull/186))
- `bin_spikes` output shape changed from `(N, T)` to `(T, N)` and default `dtype` changed from `np.float32` to `np.int32` ([#170](https://github.com/neuro-galaxy/torch_brain/pull/170))
- Renamed `torch_brain.models.CaPOYO` to `torch_brain.models.CalciumPOYOPlus` ([#198](https://github.com/neuro-galaxy/torch_brain/pull/198))
- Removed `hydra-core` as a core dependency ([#211](https://github.com/neuro-galaxy/torch_brain/pull/211))
- POYO API changes ([#206](https://github.com/neuro-galaxy/torch_brain/pull/206)):
    - Constructor argument `readout_spec` replaced with `dim_out`
    - Tokenizer no longer produces query tokens. This can be produced by the dataset (see examples/poyo)
    - `POYO.load_pretrained` now requires just providing the checkpoint path.
    - Removed `poyo_mp()` function

### Fixed
- Fixed `MultiTaskDecodingStitchEvaluator` caching predictions under the wrong sequence index when batch samples have non-overlapping readout types. ([#175](https://github.com/neuro-galaxy/torch_brain/pull/175))


## [0.1.1] - 2026-03-01
### Added
- Added `keep_files_open` flag in `Dataset` ([#88](https://github.com/neuro-galaxy/torch_brain/pull/88))
- Added `SinusoidalTimeEmbedding` ([#97](https://github.com/neuro-galaxy/torch_brain/pull/97))
- Added `eps` parameter to `bin_spikes` to improve numerical stability when computing bin boundaries. ([#160](https://github.com/neuro-galaxy/torch_brain/pull/160))
- Added `max_spikes` parameter to `bin_spikes`. ([#158](https://github.com/neuro-galaxy/torch_brain/pull/158))
- Added `torch_brain.dataset` module. ([#154](https://github.com/neuro-galaxy/torch_brain/pull/154))

### Changed
- Refactor `RotaryEmbedding` to `RotaryTimeEmbedding` ([#97](https://github.com/neuro-galaxy/torch_brain/pull/97))

### Fixed
- `InfiniteVocabEmbedding.extend_vocab`: fix incorrect device behavior ([#99](https://github.com/neuro-galaxy/torch_brain/pull/99))
- `InfiniteVocabEmbedding.load_state_dict`: fix inplace modification for source vocab ([#148](https://github.com/neuro-galaxy/torch_brain/pull/148))
- `bin_spikes`: fix bug when start time is non-zero ([#160](https://github.com/neuro-galaxy/torch_brain/pull/160))
- Fixed `task_emb` size in CaPOYO and POYO+ to account for 1-indexed modality IDs, which caused an index error when accessing the last-registered modality. ([#174](https://github.com/neuro-galaxy/torch_brain/pull/174))


## [0.1.0] - 2025-03-26
### Added
- Added a method to resolve weights based on interval membership of timestamps. ([#31](https://github.com/neuro-galaxy/torch_brain/pull/31))
- Added multitask decoder taxonomy. ([#8](https://github.com/neuro-galaxy/torch_brain/pull/8))
- Added stitching callback that takes care of stitching. ([#16](https://github.com/neuro-galaxy/torch_brain/pull/16))

### Changed
- Update workflow to use ubuntu-latest instances from github actions. ([#8](httpps://github.com/neuro-galaxy/torch_brain/pull/8))
- Simplify Dataset interface by removing the `include` dictionnary and allowing to directly load selection from a configuration file. ([#10](https://github.com/neuro-galaxy/torch_brain/pull/10))
- Sampling intervals are now represented as `Interval` objects. ([#11](https://github.com/neuro-galaxy/torch_brain/pull/11))
- `session_id` was being used for multiple purposes, and was not consistent with the data model. Replace `session_id` with `recording_id` where `recording_id` = `brainset/session`. ([#15](https://github.com/neuro-galaxy/torch_brain/pull/15))
- Improved validation metrics computation by implementing caching and stitching of predictions. ([#16](https://github.com/neuro-galaxy/torch_brain/pull/16))
- Enhanced data sampling with distributed capabilities and sequence tracking. ([#16](https://github.com/neuro-galaxy/torch_brain/pull/16))
- Updated attention layers to simplify interface and support both forward and forward_varlen modes. ([#16](https://github.com/neuro-galaxy/torch_brain/pull/16))
- Replaced Decoder enum with registry system to track different modality specifications. ([#16](https://github.com/neuro-galaxy/torch_brain/pull/16))
- Added eval_interval to config files as an optional field used to specify the interval for evaluation. ([#30](https://github.com/neuro-galaxy/torch_brain/pull/30))

### Fixed
- Fixed memory issues during validation by implementing a cache flushing mechanism. ([#16](https://github.com/neuro-galaxy/torch_brain/pull/16))
- Fixed a bug in `InfiniteVocabEmbedding` where duplicate words cause the model to fail silently. ([#9](https://github.com/neuro-galaxy/torch_brain/pull/9))
- Fixed a bug in stitcher logic when caching is enabled. ([#34](https://github.com/neuro-galaxy/torch_brain/pull/34))
