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
- **Phase 4C: Selective Response RL Environment**
  - Binary decision-making task (respond/ignore to acoustic events)
  - Configurable target materials and distractor materials
  - Asymmetric reward structure (TP/TN/FP/FN penalties)
  - Precision/Recall/F1/Accuracy metrics built-in
  - 2x2 confusion matrix tracking
  - Target probability control for class imbalance
  - Baseline agents: random, always-respond, never-respond
  - PyTorch training example with REINFORCE
  - 27 unit tests
  - Module: `physonic/envs/selective_response.py`

- **Phase 4B: Sound Classification RL Environment**
  - Material classification task (steel, glass, wood, rubber)
  - Discrete 4-class action space
  - Binary reward: +1.0 correct, 0.0 incorrect
  - Confusion matrix tracking built-in
  - Position/surface randomization options
  - CPU-friendly design (no pre-trained models)
  - Baseline agents: random (~25%), heuristic (~27%)
  - 21 unit tests
  - Module: `physonic/envs/classification.py`

- **Phase 4A: Sound Localization RL Environment**
  - Azimuth prediction task (continuous action space)
  - Angular error reward: `-|pred - true| / 180`
  - Mel-spectrogram or raw audio observations
  - Integration with physics simulation and modal synthesis
  - Seed reproducibility and Gymnasium API compliance
  - 41 unit tests
  - Module: `physonic/envs/localization.py`
  - Supporting modules: `preprocessing.py`, `rewards.py`, `base.py`

- **Phase 3B: Multi-Camera Video Rendering**
  - Camera configuration system (CameraConfig, CameraPreset)
  - 4 camera modes: fixed (front/side/top), multi-camera, tracking, egocentric
  - MuJoCo offscreen rendering with mujoco.Renderer
  - Post-hoc rendering from telemetry replay
  - MP4 encoding with H.264 (automatic codec fallback)
  - Standalone CLI tool: `python3 -m physonic.render_video`
  - Spec reconstruction from episode names
  - Dataset export integration with synchronized video
  - Parquet schema updated with video path columns
  - 73 unit tests (camera: 28, encoder: 17, renderer: 10, reconstruction: 18)
  - Modules: `physonic/render/camera.py`, `renderer.py`, `encoder.py`

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

- **v1.0.0** - Initial release with Phases 1-4C complete
  - Full physics-to-audio pipeline
  - Spatial audio and room acoustics
  - Multi-camera video rendering
  - LeRobotDataset export
  - Three RL environments (localization, classification, selective response)
  - 274 unit tests
- **v0.1.0** - Internal alpha (Phase 1 only)
