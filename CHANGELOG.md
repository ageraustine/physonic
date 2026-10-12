# Changelog

All notable changes to Physonic will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive README.md for GitHub/Hugging Face
- CONTRIBUTING.md with development guidelines
- GitHub issue templates (bug report, feature request)
- Pull request template
- GitHub Actions CI/CD workflow
- Organized outputs/ directory structure
- Helper script for organizing scattered outputs
- Example.py tutorial script
- MIT License

## [1.0.0] - 2025-01-XX

### Added
- **Phase 3A: LeRobotDataset Export**
  - Parquet-based dataset format (audio + state + ground truth)
  - Episode loading from flat and subdirectory formats
  - Audio chunking at configurable fps (default 30 fps)
  - Telemetry downsampling from 5kHz to target fps
  - State vector construction (4 floats single-object, 6N+2 multi-object)
  - FLAC audio compression
  - CLI tool: `python3 -m physonic.export_dataset`
  - 13 unit tests

- **Phase 2C: Room Acoustics**
  - Image-source method for early reflections (order 2-3)
  - Schroeder-Moorer reverberator (4 comb + 2 allpass filters)
  - RT60-based gain tuning
  - 17 predefined room templates (warehouses, offices, data centers, etc.)
  - Integration before HRTF rendering for spatial realism
  - 28 unit tests

- **Phase 2B: Multi-Object Physics**
  - Multi-object MuJoCo scenes (arbitrary object combinations)
  - Per-pair contact tracking with object labels
  - Object-object collision synthesis
  - 7 predefined scenarios (stack_collapse, sphere_cascade, etc.)
  - Object registry with materials and initial states
  - 17 unit tests

- **Phase 2A: Spatial Audio**
  - HRTF-based binaural rendering (MIT KEMAR dataset)
  - Distance attenuation (1/r law)
  - Nearest-neighbor HRTF interpolation (5° resolution)
  - ITD/ILD encoding for 3D sound localization
  - 44 unit tests

- **Phase 1: Core Pipeline**
  - MuJoCo physics simulation (5 kHz, 0.0002s timestep)
  - Modal synthesis (impacts + friction)
  - Event extraction from force traces
  - Material property database (steel, glass, wood, rubber)
  - Single-object scenarios (drop, slide)
  - Contact force telemetry export
  - 29 unit tests

### Fixed
- None yet

### Changed
- Reorganized outputs into single `outputs/` directory with subdirectories
- Updated all docstrings to reflect new output structure
- Improved .gitignore to preserve directory structure

### Deprecated
- None

### Removed
- Scattered `outputs_*` directories (consolidated into `outputs/`)

### Security
- None

---

## Version History

- **v1.0.0** - Initial release with Phases 1-3A complete
- **v0.1.0** - Internal alpha (Phase 1 only)
