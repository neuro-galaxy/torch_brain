# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2024-06-17
### Added
- Added a `domain_start` attribute to the `RegularTimeSeries` object to simplify the creation of the domain.
- Added an automated way of resolving `domain` for `Data` objects by infering it from
the domains of its attributes.
- Added documentation.
- Added special keys with the `_domain` suffix. These keys are exluded from `add_split_mask` and `_check_for_data_leakage`.
- Added warning when `timestamps` or `start` and `end` are not in `np.float64` precision.
- Added `materialize` method to lazy objects to load them directly to memory.

### Changed
- Changed slicing behavior in the `RegularTimeSeries` to make it more consistent with the `IrregularTimeSeries` object.
- Changed `repr` method of all objects to exclude split masks and `domain` attributes.

### Deprecated
- Deprecated `trials` as a special key that is not checked for data leakage.

### Fixed
- Fixed a bug where `absolute_start` was not saved to hdf5 files.

## [0.1.0] - 2024-06-11
### Added
- Initial release of the package.