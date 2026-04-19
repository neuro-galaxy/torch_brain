# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Fixed
- Fixed `RegularTimeSeries.slice` and `LazyRegularTimeSeries.slice` returning incorrect domain when slicing entirely outside the data domain. With `reset_origin=False`, the returned empty slice now has its domain clamped to the nearest boundary of the original domain (i.e., `[domain.start, domain.start]` when slicing before the data, or `[domain.end, domain.end]` when slicing after). With `reset_origin=True`, the domain is shifted relative to the slice start, yielding `[0, 0]` in both cases.

### Changed

### Removed


## [0.1.4] - 2026-03-25

### Added
- Added `Data.has_nested_attribute()`. ([#42](https://github.com/neuro-galaxy/temporaldata/pull/42))
- Added `Data.set_nested_attribute()`. ([#86](https://github.com/neuro-galaxy/temporaldata/pull/86))
- Added `Data.load()`. ([#56](https://github.com/neuro-galaxy/temporaldata/pull/56))
- Added `Data.save()`. ([#88](https://github.com/neuro-galaxy/temporaldata/pull/88))
- Added `Data.file` property, `Data.close()`, and context manager support for lazy-loaded data. ([#95](https://github.com/neuro-galaxy/temporaldata/pull/95))
- Added `Interval.subdivide()` method for fixed-duration subdivision of intervals. ([#63](https://github.com/neuro-galaxy/temporaldata/pull/63)) and ([#80](https://github.com/neuro-galaxy/temporaldata/pull/80))
- Added lazy loading support for nested `Data` objects in `Data.from_hdf5`. ([#62](https://github.com/neuro-galaxy/temporaldata/pull/62))
- Added benchmarking suite. ([#100](https://github.com/neuro-galaxy/temporaldata/pull/100))
- Added `eps` parameter to `RegularTimeSeries.slice` to handle numerical instability. ([#106](https://github.com/neuro-galaxy/temporaldata/pull/106))

### Fixed
- Fixed `RegularTimeSeries.slice` not taking the last point when the start is not aligned with timestamps, and improved numerical stability with a default `eps=1e-9`. ([#106](https://github.com/neuro-galaxy/temporaldata/pull/106))
- Fixed `RegularTimeSeries.slice` not updating the `domain` attribute, leading to incorrect `timestamps` resolution after slicing. ([#39](https://github.com/neuro-galaxy/temporaldata/pull/39))
- Fixed `Data.materialize` not loading domain information from the file. ([#43](https://github.com/neuro-galaxy/temporaldata/pull/43))
- Fixed `IrregularTimeSeries` domain setter to validate that the domain is a valid, non-overlapping, and sorted `Interval`. ([#64](https://github.com/neuro-galaxy/temporaldata/pull/64))
- Fixed `Interval.select_by_interval` edge case with point intervals. ([#111](https://github.com/neuro-galaxy/temporaldata/pull/111))
- Fixed type errors caught by type-checking harness. ([#113](https://github.com/neuro-galaxy/temporaldata/pull/113))

### Changed
- Changed minimum Python version to 3.10. ([#93](https://github.com/neuro-galaxy/temporaldata/pull/93))
- Optimized performance of `Interval.coalesce()`. ([#97](https://github.com/neuro-galaxy/temporaldata/pull/97))
- Made `temporaldata.data.serialize` private (`temporaldata.data._serialize`). ([#92](https://github.com/neuro-galaxy/temporaldata/pull/92))
- Split `temporaldata.py` into separate module files. ([#58](https://github.com/neuro-galaxy/temporaldata/pull/58))
- Performance improvements to numpy operations. ([#44](https://github.com/neuro-galaxy/temporaldata/pull/44))
- Optimized interval operations (`difference`, `__and__`, `__or__`) with vectorized implementations and improved edge case handling. ([#102](https://github.com/neuro-galaxy/temporaldata/pull/102)) and ([#111](https://github.com/neuro-galaxy/temporaldata/pull/111))
- Changed `"auto"` domain for `RegularTimeSeries` to have no impact when doing `rts.slice(rts.domain.start[0], rts.domain.end[-1])`. ([#109](https://github.com/neuro-galaxy/temporaldata/pull/109))

### Deprecated
- Started deprecation of `set_train_domain`, `set_valid_domain`, and `set_test_domain` methods in `Data`. ([#47](https://github.com/neuro-galaxy/temporaldata/pull/47))
- Started deprecation of `_check_for_data_leakage` method in `Data`. ([#47](https://github.com/neuro-galaxy/temporaldata/pull/47))

### Removed
- Removed `add_split_mask` method from `Data`, `Interval`, `IrregularTimeSeries`, and `RegularTimeSeries`. ([#47](https://github.com/neuro-galaxy/temporaldata/pull/47))
- Removed `allow_split_mask_overlap` method from `Interval`. ([#47](https://github.com/neuro-galaxy/temporaldata/pull/47))
- Removed `RegularTimeSeries.timekeys()` as it was dead code. ([#112](https://github.com/neuro-galaxy/temporaldata/pull/112))
- Removed `LazyArrayDict.load()` and `LazyIrregularTimeSeries.load()` (these performed materialization). Use `.materialize()` instead. ([#114](https://github.com/neuro-galaxy/temporaldata/pull/114))


## [0.1.3] - 2025-03-21
### Added
- Added `__iter__` method to `Interval` to iterate over the intervals. ([#36](https://github.com/neuro-galaxy/temporaldata/pull/36))

### Fixed
- Fixed a bug where `Interval` unions fails when two intervals share the same start and end. ([#32](https://github.com/neuro-galaxy/temporaldata/pull/32))
- Fixed a bug where `Data.add_split_mask` does not propagate to nested `Data` objects. ([#32](https://github.com/neuro-galaxy/temporaldata/pull/32))
- Fixed a bug where `Interval.dilate` fails when the interval is empty. ([#35](https://github.com/neuro-galaxy/temporaldata/pull/35))

### Removed
- Removed `trials` as a special key that is not checked for data leakage. ([#32](https://github.com/neuro-galaxy/temporaldata/pull/32))

## [0.1.2] - 2025-01-22
### Added
- Added documentation. ([#24](https://github.com/neuro-galaxy/temporaldata/pull/24), [#25](https://github.com/neuro-galaxy/temporaldata/pull/25), [#26](https://github.com/neuro-galaxy/temporaldata/pull/26))
- Added LICENSE file. ([#29](https://github.com/neuro-galaxy/temporaldata/pull/29))

### Changed
- Relaxed the requirements for `numpy`, `pandas`, and `h5py`. ([#27](https://github.com/neuro-galaxy/temporaldata/pull/27))

## [0.1.1] - 2024-11-11
### Added
- Added `set_train_domain`, `set_valid_domain`, and `set_test_domain` methods to `Data` to set the domain and split masks at once. ([#21](https://github.com/neuro-galaxy/temporaldata/pull/21))

### Changed
- Changed the `keys` method to `keys()` to be consistent with other packages. ([#22](https://github.com/neuro-galaxy/temporaldata/pull/22))

### Fixed
- Fixed a bug where a `LazyData` object is instanitated, but the class does not exist, and `Data` should be used instead. ([#17](https://github.com/neuro-galaxy/temporaldata/pull/17))
- Fixed a bug where `is_dijoint` calls `sort` incorrectly causing an error when evaluating unsorted intervals. ([#20](https://github.com/neuro-galaxy/temporaldata/pull/20))

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
