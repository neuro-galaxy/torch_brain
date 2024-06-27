# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Add package version to `BrainsetDescription`.

### Removed
- Removed the dataset_builder class. Validation can be done through other means.

### Changed
- Renamed `dandiset` to `brainset`.
- Replaced `sortset` with `device`.
- Updated snakemake pipeline to process one file at a time.

## [0.1.0] - 2024-06-11
### Added
- Initial release of the package.