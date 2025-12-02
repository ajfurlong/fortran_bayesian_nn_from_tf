# Changelog

## [1.1.1] - 2025-12-02
### Added
- N/A

### Changed
- More robust handling of real precision
- Main remains unchanged

### Fixed
- Precision handling was pretty brittle, it should be strong logic now

## [1.1.0] - 2025-01-18
### Added
- Verification toy problem "nonlinear regression" and results
- New subroutine save_verification_data() added to metrics_module.f90 to allow for data analysis outside of Fortran

### Changed
- HDF5 processing updated to detect if file dataset is F32 or F64 (TensorFlow model.save() defaults to F32)
- Casts HDF5 read-in datasets to correct precision if different than the compiled
- Renamed activation functions "relu" and "elu" -> "relu_fn" and "elu_fn"

### Fixed
- Corrected support for double precision

## [1.0.0] - 2024-12-19
### Added
- Initial release.