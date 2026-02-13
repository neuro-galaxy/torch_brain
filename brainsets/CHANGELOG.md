# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added normalization schemes for sex, age, and species in `SubjectDescription` ([#78](https://github.com/neuro-galaxy/brainsets/pull/78)).
- Added generic data extraction helpers in `mne_utils` to handle MNE Raw objects ([#78](https://github.com/neuro-galaxy/brainsets/pull/78)).
- Enriched `s3_utils` with additional functionalities to get data from public buckets ([#78](https://github.com/neuro-galaxy/brainsets/pull/78)).

### Removed

### Changed
- Suppress INFO logs from ray when calling `brainsets prepare` ([#70](https://github.com/neuro-galaxy/brainsets/pull/70))

## [0.2.0] - 2025-12-24
### Added
- New brainset pipeline: `allen_visual_coding_ophys_2016` ([#16](https://github.com/neuro-galaxy/brainsets/pull/16)).
- Added `STEREO_EEG` to `RecordingTech` enum ([#36](https://github.com/neuro-galaxy/brainsets/pull/36)).
- Added `BrainsetPipeline` definition with a `brainset.runner` module to run `BrainsetPipeline` in parallel ([#37](https://github.com/neuro-galaxy/brainsets/pull/37)).
- Added support for PEP723-style inline metadata blocks to specify Python version and dependencies in a `pipeline.py` file ([#62](https://github.com/neuro-galaxy/brainsets/pull/62)).
- New brainset pipeline: `kemp_sleep_edf_2013` ([#54](https://github.com/neuro-galaxy/brainsets/pull/54)).

### Changed
- Fixed issue with "test" session being overwritten in the `perich_miller_population_2018` pipeline ([#25](https://github.com/neuro-galaxy/brainsets/pull/25)).
- Updated all existing pipelines with `BrainsetPipeline` and dependencies in metadata block ([#37](https://github.com/neuro-galaxy/brainsets/pull/37) and [#62](https://github.com/neuro-galaxy/brainsets/pull/62)).


## [0.1.3] - 2025-10-27
### Added
- Added support for python 3.12 and 3.13 ([#14](https://github.com/neuro-galaxy/brainsets/pull/30)).

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
