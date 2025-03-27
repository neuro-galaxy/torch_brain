# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2025-03-27
### Added
- Added package version to `BrainsetDescription`.
- Added pipeline for allen_visual_coding_ophys_2016 ([#4](https://github.com/neuro-galaxy/brainsets/pull/4)).
- Added split functions to split variable number of epochs in train/validation/test ([#4](https://github.com/neuro-galaxy/brainsets/pull/4)).
- Added allen-related taxonomy. ([#4](https://github.com/neuro-galaxy/brainsets/pull/4)).
- Added unit tests for all enums to check for duplicates. ([#8](https://github.com/neuro-galaxy/brainsets/pull/8)).

### Removed
- Removed the dataset_builder class. Validation can be done through other means.
- Removed multitask_readout. ([#8](https://github.com/neuro-galaxy/brainsets/pull/8))

### Changed
- Fixed issue in snakemake where checkpoints are global variables  ([#4](https://github.com/neuro-galaxy/brainsets/pull/4)).
- Renamed `dandiset` to `brainset`.
- Replaced `sortset` with `device`.
- Updated snakemake pipeline to process one file at a time.
- Made tests that require mne to be skipped if mne is not installed.

## [0.1.0] - 2024-06-11
### Added
- Initial release of the package.