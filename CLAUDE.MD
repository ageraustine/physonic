# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Physonic is a procedural physics-acoustic synthesis pipeline that generates realistic contact sounds from MuJoCo simulations. It simulates physical interactions (drops, slides, multi-object collisions) and synthesizes audio using modal synthesis driven by contact telemetry rather than sample playback.

**End Goal**: Train robots on complex acoustic perception tasks (sound localization, sound classification, selective response) in simulated commercial environments - data centers, warehouses, supermarkets, etc. Physics fidelity is critical for commercial viability: robots must learn from physically-accurate acoustic training data that transfers to real-world deployment.

**Current Status**: Phase 2C Complete - Core physics-to-audio pipeline with spatial audio (binaural HRTF), multi-object physics, and room acoustics (image-source + Schroeder reverb). Next: LeRobotDataset export and environment templates (Phase 3).

## Setup: Using Local .claude/ Configuration

This project uses project-specific skills and configuration in `.claude/`. To ensure Claude Code uses **only** the local configuration (not global `~/.claude`), you must set `CLAUDE_CONFIG_DIR`:

### Option 1: direnv (Recommended)
```bash
# Install direnv if needed
brew install direnv  # macOS
# or: apt install direnv  # Linux

# Enable direnv in your shell (~/.bashrc or ~/.zshrc)
eval "$(direnv hook bash)"  # for bash
eval "$(direnv hook zsh)"   # for zsh

# Allow the .envrc in this project
cd /path/to/physonic_mvp
direnv allow

# Future sessions will automatically use local .claude/
```

### Option 2: Per-session
```bash
CLAUDE_CONFIG_DIR="$PWD/.claude" claude
```

### Option 3: Shell alias
```bash
# Add to ~/.bashrc or ~/.zshrc
alias claude-local='CLAUDE_CONFIG_DIR="$PWD/.claude" claude'

# Then use:
claude-local
```

**Important Notes**:
- Without `CLAUDE_CONFIG_DIR`, the local `.claude/` will **layer on top** of global `~/.claude` (project settings override global, but both are loaded)
- With `CLAUDE_CONFIG_DIR` set, **only** the local `.claude/` is used
- You may need to log in again the first time (credentials are per config dir)
- Use absolute path (`$PWD/.claude`) not relative (`./.claude`) to avoid issues when Claude Code changes directories

**Test Coverage**: 89 tests passing
- `tests/test_contact/`: Contact physics & synthesis tests
- `tests/test_spatial/`: 44 spatial audio tests (HRTF, ITD/ILD validation)
- `tests/test_acoustics/`: 28 room acoustics tests (1 skipped for RT60 calibration)

**Key Features**:
- ✅ MuJoCo physics simulation (5 kHz)
- ✅ Modal synthesis (impacts + friction)
- ✅ Multi-object scenarios (7 predefined)
- ✅ Binaural spatial audio (HRTF-based)
- ✅ Room acoustics (17 room templates)
- ✅ LeRobotDataset export (Phase 3A)
- ⏳ Video rendering (Phase 3B)
- ⏳ RL environments (Phase 4)

## Directory Structure

```
physonic/
├── __init__.py
├── contact/                  # Physics simulation & audio synthesis
│   ├── __init__.py
│   ├── rig.py               # MuJoCo scene builder, simulation, event extraction
│   ├── modal_synth.py       # Modal synthesis (impacts + friction)
│   ├── materials.py         # Material property database (physics + acoustics)
│   └── scenarios.py         # Predefined multi-object scenarios (7 scenarios)
├── spatial/                  # Binaural spatial audio (Phase 2A)
│   ├── __init__.py
│   ├── hrtf.py              # HRTF loading (MIT KEMAR dataset)
│   ├── microphone.py        # BinauralMicrophone model
│   └── renderer.py          # Spatial audio rendering (distance + HRTF)
├── acoustics/                # Room acoustics (Phase 2C)
│   ├── __init__.py
│   ├── room.py              # RoomSpec, image-source method, Schroeder reverb
│   └── rooms.py             # 17 predefined room templates (warehouses, offices, etc.)
├── export/                   # LeRobotDataset export (Phase 3A)
│   ├── __init__.py
│   ├── episode.py           # Episode dataclass, load_episode() for flat/subdirectory
│   ├── writer.py            # LeRobotWriter, audio chunking, telemetry downsampling
│   └── schemas.py           # PyArrow schema definitions
├── run_demo.py              # Main CLI: single-object scenarios with spatial + reverb
├── run_demo_multi.py        # Multi-object scenarios CLI
├── export_dataset.py        # Dataset export CLI
└── verify.py                # Output validation & spectrogram generation

tests/
├── test_contact/            # Contact physics & synthesis tests
│   ├── test_materials.py
│   ├── test_modal_synth.py
│   ├── test_rig.py
│   └── test_multi_object.py # Multi-object physics tests (17 tests)
├── test_spatial/            # Spatial audio tests (44 tests)
│   ├── test_hrtf.py
│   ├── test_microphone.py
│   └── test_renderer.py
├── test_acoustics/          # Room acoustics tests (28 tests)
│   └── test_room.py
└── test_export/             # Dataset export tests (13 tests)
    ├── test_episode.py
    └── test_writer.py

assets/
└── hrtf/
    └── test_kemar/          # MIT KEMAR HRTF dataset (5° resolution)
```

## Local Skills (Project-Specific)

**Location**: `.claude/skills/` (local to this project, not global)

These are custom domain experts for Physonic development. Invoke them when working in their specific areas:

### `/spatial-audio-engineer`
**When to use**: Phase 2A-2C spatial audio implementation, HRTF debugging, room acoustics
- Expert in HRTF binaural rendering (MIT KEMAR dataset)
- Room acoustics: image-source method, Schroeder reverb, RT60 tuning
- ITD/ILD validation and debugging
- Microphone array design

### `/dataset-engineer`
**When to use**: Phase 3A dataset export, Parquet schema design, Hugging Face workflows
- LeRobotDataset format and structure
- Parquet schema design for multi-modal data (audio + video + state)
- Hugging Face Hub upload and dataset cards
- Timestamp synchronization (44.1kHz audio, 30fps video, 5kHz physics)

### `/rl-engineer`
**When to use**: Phase 4 RL environment implementation, reward shaping, training debugging
- Gymnasium API and observation/action space design
- Reward shaping for acoustic perception tasks
- PPO/SAC policy training with Stable-Baselines3
- Multi-modal RL (audio + vision fusion)
- Audio feature extraction (mel-spectrograms, MFCCs)

## Commands

**Note**: All outputs are organized in the `outputs/` directory with subdirectories for different experiment types.

### Single-object scenarios (mono or binaural)
```bash
# Mono audio only
python3 -m physonic.run_demo --out outputs/mono --seed 0

# Binaural spatial audio
python3 -m physonic.run_demo --spatial --hrtf-path assets/hrtf/test_kemar --out outputs/spatial

# Spatial + room acoustics (preset)
python3 -m physonic.run_demo --spatial --room warehouse_large --out outputs/reverb

# Spatial + room acoustics (custom)
python3 -m physonic.run_demo --spatial --room-dim 30,20,8 --rt60 2.5 --absorption 0.15 --out outputs/custom
```
Generates 8 audio clips (4 drop + 4 slide scenarios across steel/glass/wood/rubber surfaces) with telemetry and event logs. Suffix: `_binaural.wav` for spatial, `_binaural_reverb.wav` for spatial+room.

### Multi-object scenarios
```bash
# Available scenarios: two_sphere_collision, stack_collapse, sphere_cascade,
#                      tumble_box, scatter_pile, domino_chain, rolling_cylinders
python3 -m physonic.run_demo_multi --scenario stack_collapse --out outputs/multi --seed 42

# With spatial audio
python3 -m physonic.run_demo_multi --scenario sphere_cascade --spatial --hrtf-path assets/hrtf/test_kemar --out outputs/multi_spatial
```

### Verify outputs
```bash
python3 -m physonic.verify --out outputs/mono
```
Runs sanity checks and generates `spectrograms.png` visualization.

### Export to LeRobotDataset
```bash
# Export all scenarios from outputs/mono directory
python3 -m physonic.export_dataset --input outputs/mono --output datasets/physonic_v1

# Export specific scenarios
python3 -m physonic.export_dataset --input outputs/mono --output datasets/test \
  --scenarios drop_steel_on_steel,slide_wood_on_glass

# Custom frame rate and chunk size
python3 -m physonic.export_dataset --input outputs/spatial --output datasets/physonic_60fps \
  --fps 60 --chunk-size 500
```
Converts simulation outputs to Parquet format with synchronized audio chunks (FLAC).

**Organized directory structure**:
```
outputs/
├── mono/              # Mono audio (8 scenarios)
├── spatial/           # Binaural spatial audio
├── reverb/            # Spatial + room acoustics
├── multi/             # Multi-object scenarios
└── custom/            # Custom experiments

datasets/
└── physonic_v1/
    ├── data/chunk-000.parquet      # State observations at fps (30 Hz default)
    ├── audio/binaural/*.flac       # Audio chunks (1470 samples @ 30 fps)
    └── meta/info.json              # Dataset metadata
```

### Run tests
```bash
# All tests
pytest -v

# Specific module
pytest tests/test_contact/ -v
pytest tests/test_spatial/ -v
pytest tests/test_acoustics/ -v
```

### Dependencies
Install: `mujoco numpy scipy soundfile matplotlib pytest`

Virtual environment is in `.venv/` (activate with `source .venv/bin/activate`).

## Architecture

The pipeline has four sequential stages:

### 1. Physics Simulation (`contact/rig.py`)
- **Single-object**: `build_xml(SceneSpec)` - drop (sphere from 0.5m) or slide (box, 3 m/s initial velocity)
- **Multi-object**: `build_multi_xml(MultiSceneSpec)` - arbitrary object combinations (spheres, boxes, capsules, cylinders)
- Four surface materials: steel, glass, wood, rubber (defined in `materials.py`)
- Logs at 5 kHz: normal force `fn`, tangential speed `vt`, object positions `z`
- Returns telemetry dict with smoothed force `fn_s` (20ms moving average to handle MuJoCo soft-contact micro-hopping)
- **Multi-object**: Returns per-pair telemetry `{(obj1, obj2): tele_dict}` for object-object and object-surface contacts

### 2. Event Extraction (`contact/rig.py`)
- **Single-object**: `extract_events(tele)` - converts continuous force trace into discrete impact events
- **Multi-object**: `extract_events_multi(telemetry, registry)` - per-pair event extraction with object labels
- Each event: onset time `t`, impulse `J` (integral of excess force over first 20ms), object IDs
- Filters micro-hops: rejects impacts with `J/mass < 0.3 m/s` equivalent velocity
- Object-object collisions use reduced mass: `m_eff = (m1 * m2) / (m1 + m2)`

### 3. Acoustic Synthesis (`contact/modal_synth.py`)
**Impact synthesis** (`render_impact`):
- Modal synthesis: sum of decaying sinusoids (resonant modes)
- Impulse `J` sets mode amplitudes
- Material sets frequencies (`f0`, `ratios`), damping (`t60`, `damp_slope`)
- Hertz contact time `tc` (computed from impulse, mass) low-passes excitation (softer materials sound duller)
- Both surface and object materials contribute (weighted 1.0 : 0.6)
- **Multi-object**: Both colliding objects vibrate (weights 1.0 : 0.7)

**Friction synthesis** (`render_friction`):
- Envelope: `sqrt(normal_force) × sliding_speed`
- Excitation: broadband noise + filtered modal ringing + stick-slip modulation
- Modes broadened (`FRIC_BW = 6.0`) vs. clean impact ringing

Mode frequencies are jittered per object (`sample_modes`) to avoid identical-sounding instances.

**Output**: Mono audio array (44.1 kHz, float32)

### 4A. Room Acoustics (`acoustics/room.py`) - OPTIONAL
Applied **before** spatial rendering for maximum realism (each reflection ray gets spatially filtered).

**Early reflections** (image-source method):
- `compute_image_sources()` - Mirror-box reflections up to order 2-3 (26-124 image sources)
- Geometric distance attenuation (1/r) + wall absorption compounds per reflection
- Returns discrete reflection times/amplitudes

**Late diffuse reverb** (Schroeder-Moorer):
- 4 parallel comb filters (echo density) + 2 serial allpass filters (smoothing)
- RT60-based gain tuning: `g = 10^(-3 * delay / RT60)`
- Room volume scales delay line lengths

**Presets**: 17 room templates (warehouses, data centers, supermarkets, offices, hallways, industrial, special)

### 4B. Spatial Rendering (`spatial/renderer.py`) - OPTIONAL
**Distance attenuation**:
- `1 / max(distance, 0.1)` to avoid infinity at source

**HRTF filtering** (binaural):
- `load_kemar_hrtf()` - MIT KEMAR dataset (5° resolution, 128-sample impulses)
- `interpolate_hrtf()` - Nearest-neighbor lookup by azimuth/elevation
- `convolve_hrtf()` - FFT-based convolution (scipy.signal.fftconvolve)
- Encodes ITD (±0.65 ms @ ±90°) and ILD (frequency-dependent level differences)

**Output**: Stereo audio array (44.1 kHz, float32, shape (N, 2))

## Material System (`materials.py`)

Each `Material` has:
- **Physics params**: `friction`, `solref` (MuJoCo contact stiffness/damping)
- **Acoustic params**: `f0` (fundamental), `ratios` (mode harmonics), `t60` (decay), `gain` (loudness), `tc_ref` (contact brightness)

Key insight: Surface material affects contact physics; object material affects acoustic timbre. In simulation, only surface physics params are used. Both materials contribute to audio synthesis.

Mode ratio sets:
- `_BAR`: Euler-Bernoulli free-free bar (1, 2.756, 5.404, ...)  — used for steel, glass
- `_WOOD`: Denser inharmonic spectrum (1, 1.9, 3.1, ...) — used for wood, rubber

## Data Flow

### Single-object pipeline
```
SceneSpec → build_xml → MuJoCo → telemetry (fn, vt, z)
                                       ↓
                              extract_events (t, J)
                                       ↓
                    sample_modes (per-object randomization)
                                       ↓
         render_impact + render_friction → mono audio
                                       ↓
                        [OPTIONAL] room acoustics (early + late reverb)
                                       ↓
                        [OPTIONAL] spatial rendering (distance + HRTF)
                                       ↓
                                  stereo .wav
```

### Multi-object pipeline
```
MultiSceneSpec → build_multi_xml → MuJoCo → per-pair telemetry {(obj1, obj2): tele}
                                                    ↓
                                       extract_events_multi (per-pair events with labels)
                                                    ↓
                               sample_modes (per-material mode banks)
                                                    ↓
            render_impact (both objects vibrate) → mono audio → [room + spatial] → stereo .wav
```

Outputs per clip:
- `{name}.wav` or `{name}_binaural.wav` or `{name}_binaural_reverb.wav` (44.1 kHz, float32)
- `{name}_telemetry.npz` (t, fn, vt, z arrays - single object) or per-pair telemetry (multi-object)
- `{name}_events.json` (impact list: t, J, obj1, obj2, contact_pos)

## Known Limitations

1. **Acoustic params are approximations**: Mode frequencies use ratio templates with jitter, not mesh-derived eigenfrequencies. Good enough for plausibility, not for material discrimination research.

2. **Contact physics ignores object material**: Only surface material affects friction/stiffness in MuJoCo. Object material only affects audio timbre.

3. **Micro-hopping**: MuJoCo soft contacts cause resting bodies to flicker on/off contact. Mitigated by:
   - Smoothing `fn` with 20ms moving average for friction
   - Rejecting low-impulse events (< 0.3 m/s equivalent)
   - Boosting damping ratio for sliding scenarios

4. **Schroeder RT60 calibration**: Reverb decays faster than specified RT60. Core functionality works, but gain formula needs fine-tuning for accurate decay times.

5. **HRTF resolution**: 5° angular resolution with nearest-neighbor interpolation. Sufficient for perception training, but VBAP or spherical harmonics would be smoother.

6. **No video rendering yet**: Planned for Phase 3B (multi-camera synchronized with audio).

7. **No LeRobotDataset export**: Planned for Phase 3A.

## Code Organization

**Entry points**:
- `run_demo.py`: Single-object scenarios with spatial + room acoustics
- `run_demo_multi.py`: Multi-object scenarios (7 predefined scenarios)
- `verify.py`: Post-hoc validation and spectrogram visualization

**Core modules**:
- `contact/rig.py`: MuJoCo scene builder (single + multi-object), simulation loop, event extraction
- `contact/modal_synth.py`: All audio synthesis (impacts, friction, mode banks)
- `contact/materials.py`: Material property database (physics + acoustics)
- `contact/scenarios.py`: 7 predefined multi-object scenarios

**Spatial audio**:
- `spatial/hrtf.py`: MIT KEMAR HRTF loading and interpolation
- `spatial/microphone.py`: BinauralMicrophone model
- `spatial/renderer.py`: Distance + HRTF convolution

**Room acoustics**:
- `acoustics/room.py`: RoomSpec, image-source method, Schroeder reverberator
- `acoustics/rooms.py`: 17 predefined room templates

**Dataset export**:
- `export/episode.py`: Episode dataclass, load_episode() for both flat and subdirectory formats
- `export/writer.py`: LeRobotWriter, audio chunking, telemetry downsampling
- `export/schemas.py`: PyArrow schema definitions for Parquet
- `export_dataset.py`: CLI tool for batch export

## Key Constants

- `TIMESTEP = 0.0002` (5 kHz physics, rig.py:11)
- `SR = 44100` (audio sample rate, modal_synth.py:17)
- `AMP = 0.25` (global impact gain, modal_synth.py:18)
- `FRIC_AMP = 0.012` (friction gain, modal_synth.py:19)

## Development Notes

- Scene specs use dataclass `SceneSpec` (rig.py:15) with `kind`, `surface`, `obj` material
- Object sizes: drop sphere 0.03m radius, slide box 0.05m half-extent (rig.py:30)
- Masses: drop 0.2kg, slide 0.5kg (rig.py:29)
- Audio uses soft limiter `tanh(x)` to prevent clipping (modal_synth.py:95)

## Roadmap to Commercial Robot Acoustic Perception

### ✅ Phase 1: Core Physics-to-Audio Pipeline (Complete)
- MuJoCo physics simulation with contact telemetry
- Modal synthesis (impacts + friction)
- Event extraction from force traces
- Material property database

### ✅ Phase 2A: Spatial Audio (Complete)
- HRTF-based binaural rendering (MIT KEMAR dataset)
- Distance attenuation (1/r law)
- ITD/ILD validation (44 tests passing)

### ✅ Phase 2B: Multi-Object Physics (Complete)
- Multi-object MuJoCo scenes (arbitrary combinations)
- Per-pair contact tracking
- Object-object collision handling
- 7 predefined scenarios (17 tests passing)

### ✅ Phase 2C: Room Acoustics (Complete)
- Image-source method for early reflections
- Schroeder reverberator for late diffuse reverb
- 17 predefined room templates (warehouses, offices, etc.)
- 28 tests passing

### ✅ Phase 3A: LeRobotDataset Export (Complete)
- Parquet schema for multi-modal data (audio + state + ground truth)
- Episode loading from flat and subdirectory output formats
- Audio chunking at 30 fps (1470 samples/frame @ 44.1kHz)
- Telemetry downsampling from 5kHz to 30 fps
- State vector construction (4 floats single-object, 6N+2 multi-object)
- FLAC audio chunks synchronized with Parquet rows
- CLI tool: `python3 -m physonic.export_dataset`
- 13 unit tests passing

### 🔄 Phase 3: Video Rendering & Environments (Next)
**3B: Multi-Camera Video Rendering**
- MuJoCo offscreen rendering (RGB cameras)
- Camera intrinsics/extrinsics metadata
- Synchronized with audio timeline
- MP4 encoding (H.264)

**3C: Environment Templates**
- Warehouse: forklifts, conveyor belts, package drops
- Data center: server fans, cooling systems, footsteps
- Supermarket: shopping carts, shelving, checkout beeps
- Procedural generation: randomized layouts, object placements, acoustic properties

### Phase 4: RL Environments for Acoustic Perception
**4A: Sound Source Localization**
- Gymnasium environment with audio observation space (mel-spectrograms)
- Continuous azimuth action space or discrete bins
- Angular error reward: `-|pred - true| / 180`

**4B: Sound Classification**
- Multi-class discrete action space
- Binary reward (correct/incorrect)
- Confusion matrix tracking

**4C: Selective Response**
- Binary action space (respond/ignore)
- False positive/missed detection penalties
- Precision/Recall/F1 evaluation

**4D: Multi-Modal Fusion**
- Vision + audio observation space
- Robust perception with sensor dropout
- Attention mechanism analysis

### Technical Debt to Address
- Fine-tune Schroeder RT60 decay calibration
- Replace material approximations with mesh-derived modal data
- Validate synthesis against real-world recordings (perceptual studies)
- VBAP or spherical harmonics for smoother HRTF interpolation
- USD/Omniverse export for NVIDIA Isaac Sim integration
- WebSocket/gRPC streaming for online RL training
