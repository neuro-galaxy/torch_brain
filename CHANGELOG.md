# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
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

### Fixed
- Fixed memory issues during validation by implementing a cache flushing mechanism. ([#16](https://github.com/neuro-galaxy/torch_brain/pull/16))
- Fixed a bug in `InfiniteVocabEmbedding` where duplicate words cause the model to fail silently. ([#9](https://github.com/neuro-galaxy/torch_brain/pull/9))
