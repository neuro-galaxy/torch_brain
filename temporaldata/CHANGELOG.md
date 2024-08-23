# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Fixed
- Fixed a bug where a `LazyData` object is instanitated, but the class does not exist, and `Data` should be used instead. ([#17](https://github.com/neuro-galaxy/temporaldata/pull/17))


## [0.1.1] - 2024-06-17
### Added
- Added a `domain_start` attribute to the `RegularTimeSeries` object to simplify the creation of the domain. ([#8](https://github.com/neuro-galaxy/temporaldata/pull/8))
- Added an automated way of resolving `domain` for `Data` objects by infering it from
the domains of its attributes. ([#7](https://github.com/neuro-galaxy/temporaldata/pull/7))
- Added documentation. ([#6](https://github.com/neuro-galaxy/temporaldata/pull/6))
- Added special keys with the `_domain` suffix. These keys are exluded from `add_split_mask` and `_check_for_data_leakage`. ([#2](https://github.com/neuro-galaxy/temporaldata/pull/2))
- Added warning when `timestamps` or `start` and `end` are not in `np.float64` precision. ([#5](https://github.com/neuro-galaxy/temporaldata/pull/5))
- Added `materialize` method to lazy objects to load them directly to memory. ([#3](https://github.com/neuro-galaxy/temporaldata/pull/3))

### Changed
- Changed slicing behavior in the `RegularTimeSeries` to make it more consistent with the `IrregularTimeSeries` object. ([#4](https://github.com/neuro-galaxy/temporaldata/pull/4) [#12](https://github.com/neuro-galaxy/temporaldata/pull/12))
- Changed `repr` method of all objects to exclude split masks and `domain` attributes. ([#10](https://github.com/neuro-galaxy/temporaldata/pull/10))

### Deprecated
- Deprecated `trials` as a special key that is not checked for data leakage. ([#2](https://github.com/neuro-galaxy/temporaldata/pull/2))

### Fixed
- Fixed a bug where `absolute_start` was not saved to hdf5 files. ([#9](https://github.com/neuro-galaxy/temporaldata/pull/9))

## [0.1.0] - 2024-06-11
### Added
- Initial release of the package.