# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of CUDA WSL benchmarking suite
- Automatic CUDA installation for GTX 1080 Ti (sm_61) with CUDA 12.5
- Idempotent environment setup with GPU/CPU fallbacks
- Benchmarks: PyTorch matrix multiplication, TensorFlow CNN, cuDF groupby
- Gamified leaderboard with device tracking, deltas, and status
- Support for CUDA 12.5 and 13.0 users
- Comprehensive documentation and troubleshooting

### Fixed
- CUDA detection logic to return "cpu" when nvcc not available
- Pip command to avoid empty PYTORCH_INDEX causing invalid requirements
- PyTorch GPU fallback on kernel image errors during execution
- Keras deprecation warning by replacing `input_shape` with `shape`
- TensorFlow CUDA initialization warnings by setting `TF_CPP_MIN_LOG_LEVEL=3`
- JSON corruption in leaderboard files
- Markdown rendering issues in README

### Changed
- Leaderboard structure with Device, Delta, Faster % columns
- Outdated flag for historical entries
- Contribution instructions for environment setup
- Wipe script for consistent clean installs

### Security
- No hardcoded secrets or API keys
