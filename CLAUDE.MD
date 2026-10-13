# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Physonic is a procedural physics-acoustic synthesis pipeline that generates realistic contact sounds from MuJoCo simulations. It simulates physical interactions (drops, slides, multi-object collisions) and synthesizes audio using modal synthesis driven by contact telemetry rather than sample playback.

**End Goal**: Train robots on complex acoustic perception tasks (sound localization, sound classification, selective response) in simulated commercial environments - data centers, warehouses, supermarkets, etc. Physics fidelity is critical for commercial viability: robots must learn from physically-accurate acoustic training data that transfers to real-world deployment.

**Current Status**: Phase 3C (Partial) + Phase 4C Complete - RL environment suite with sound localization, material classification, and selective response tasks, plus full multi-modal dataset generation pipeline. Phase 3C adds procedural environment generation with sound synthesis infrastructure (continuous equipment sounds, alert/beep synthesis, acoustic scene mixing) and data center template with layout generation.

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

**Test Coverage**: 342 tests passing, 4 skipped (89 envs, 73 render, 72 synthesis, 44 spatial, 28 acoustics, 21 export, 19 contact)
- `tests/test_envs/`: 89 RL environment tests (41 localization, 21 classification, 27 selective response)
- `tests/test_render/`: 73 video rendering tests (camera, encoder, renderer, reconstruction)
- `tests/test_synthesis/`: 72 sound synthesis tests (continuous sounds, alerts, mixing) - NEW
- `tests/test_spatial/`: 44 spatial audio tests (HRTF, ITD/ILD validation)
- `tests/test_acoustics/`: 28 room acoustics tests (1 skipped for RT60 calibration)
- `tests/test_export/`: 21 dataset export tests (including 5 video integration tests)
- `tests/test_contact/`: 19 contact physics & multi-object tests

**Key Features**:
- ✅ MuJoCo physics simulation (5 kHz)
- ✅ Modal synthesis (impacts + friction)
- ✅ Multi-object scenarios (7 predefined)
- ✅ Binaural spatial audio (HRTF-based)
- ✅ Room acoustics (17 room templates)
- ✅ Sound synthesis (continuous equipment sounds, alerts/beeps, mixing) - Phase 3C
- ✅ Procedural environment generation (data center template with layout algorithms) - Phase 3C
- ✅ LeRobotDataset export (Phase 3A)
- ✅ Video rendering (Phase 3B - standalone CLI + dataset integration)
- ✅ RL environments (Phase 4A - Sound Localization, Phase 4B - Sound Classification, Phase 4C - Selective Response)

## Directory Structure

**Important**: All Python files must be in `physonic/` (library code) or `scripts/` (executables). **No Python files at project root.**

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
├── render/                   # Video rendering (Phase 3B)
│   ├── __init__.py
│   ├── camera.py            # CameraConfig, CameraPreset, camera modes
│   ├── renderer.py          # VideoRenderer, MuJoCo offscreen rendering
│   └── encoder.py           # VideoEncoder, MP4 encoding (H.264)
├── synthesis/                # Sound synthesis (Phase 3C) - NEW
│   ├── __init__.py
│   ├── continuous.py        # Continuous equipment sounds (fans, servers, HVAC)
│   ├── alerts.py            # Alert/beep sounds (alarms, warnings)
│   └── mixer.py             # Multi-source mixing (impact + continuous + alerts)
├── environments/             # Environment templates (Phase 3C) - NEW
│   ├── __init__.py
│   ├── layout.py            # Grid-based procedural layout generation
│   ├── objects.py           # Static objects for MuJoCo scenes
│   └── data_center.py       # Data center environment template
├── envs/                     # RL environments (Phase 4A-4C)
│   ├── __init__.py
│   ├── base.py              # BaseAcousticEnv (shared functionality)
│   ├── localization.py      # SoundLocalizationEnv (azimuth prediction)
│   ├── classification.py    # SoundClassificationEnv (material classification)
│   ├── selective_response.py # SelectiveResponseEnv (binary respond/ignore)
│   ├── preprocessing.py     # extract_mel_spectrogram()
│   └── rewards.py           # Reward functions
└── visualization/            # Demo and visualization tools
    ├── __init__.py
    ├── recorder.py          # Episode recording (Gymnasium/HF compatible)
    └── plots.py             # Visualization utilities (waveforms, spectrograms, metrics)

scripts/                      # Executable scripts (organized by category)
├── baselines/               # Baseline agents for RL environments (5 scripts)
│   ├── baseline_localization.py     # Random/perfect agents for localization
│   ├── baseline_classification.py   # Random/heuristic agents for classification
│   ├── baseline_selective.py        # Random/always/never agents for selective response
│   ├── baseline_pytorch_localization.py  # PyTorch REINFORCE for localization
│   └── baseline_pytorch_selective.py     # PyTorch REINFORCE for selective response
├── demos/                   # Interactive demos (3 scripts)
│   ├── demo_rl_env.py              # RL environment demo (all 3 tasks)
│   ├── demo_environment_gen.py     # Procedural environment generation
│   └── generate_data_center_layouts.py  # Batch layout generation
├── training/                # PPO training infrastructure (5 scripts)
│   ├── train_ppo_data_center.py    # PPO agent training with video recording
│   ├── video_recorder_callback.py  # Evaluation video recording callback
│   ├── training_monitor_callback.py # Training rollout video callback
│   └── test_video_recording.py     # Video codec diagnostic tool
├── visualization/           # Visualization tools (2 scripts)
│   ├── view_environment.py         # 3D environment viewer
│   └── visualize_audio_observation.py  # Audio perception visualization
├── utils/                   # Utility scripts (2 scripts)
│   ├── setup_hrtf_test.py         # HRTF test dataset setup
│   └── organize_outputs.py        # Output organization tool
└── examples/                # Basic examples (1 script)
    └── example.py                  # Basic usage example

# Legacy scripts (still in physonic/ for backward compatibility)
physonic/
├── run_demo.py              # Main CLI: single-object scenarios with spatial + reverb
├── run_demo_multi.py        # Multi-object scenarios CLI
├── export_dataset.py        # Dataset export CLI
├── render_video.py          # Video rendering CLI (standalone, post-hoc from telemetry)
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
├── test_export/             # Dataset export tests (13 tests)
│   ├── test_episode.py
│   └── test_writer.py
├── test_render/             # Video rendering tests (73 tests)
│   ├── test_camera.py       # Camera configuration & presets (28 tests)
│   ├── test_encoder.py      # MP4 encoding (17 tests)
│   ├── test_renderer.py     # MuJoCo rendering (10 tests)
│   └── test_reconstruction.py # Spec reconstruction (18 tests)
├── test_synthesis/          # Sound synthesis tests (72 tests) - NEW
│   ├── test_continuous.py  # Continuous sound tests
│   ├── test_alerts.py      # Alert/beep sound tests
│   └── test_mixer.py       # Mixing tests
└── test_envs/               # RL environment tests (89 tests)
    ├── test_localization.py # Sound localization tests (41 tests)
    ├── test_classification.py # Sound classification tests (21 tests)
    └── test_selective_response.py # Selective response tests (27 tests)

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

# With video rendering (Phase 3B - in progress)
python3 -m physonic.export_dataset --input outputs/mono --output datasets/with_video \
  --include-video --video-cameras multi
```
Converts simulation outputs to Parquet format with synchronized audio chunks (FLAC) and optional video.

### Render Videos (Phase 3B)
```bash
# Single front camera
python3 -m physonic.render_video \
  --input outputs/test_video \
  --scenario drop_steel_on_glass \
  --camera front \
  --output video.mp4

# Multi-camera (front, side, top)
python3 -m physonic.render_video \
  --input outputs/test_video \
  --scenario drop_steel_on_glass \
  --camera multi \
  --output outputs/test_video/videos/

# Tracking camera (follows object)
python3 -m physonic.render_video \
  --input outputs/multi \
  --scenario cascade \
  --camera tracking \
  --track-object sphere1 \
  --output tracking.mp4

# Custom resolution and framerate
python3 -m physonic.render_video \
  --input outputs/test_video \
  --scenario drop_steel_on_glass \
  --camera front \
  --width 1280 --height 720 --fps 60 \
  --output video_hd.mp4
```
Post-hoc video rendering from saved telemetry. Reconstructs MuJoCo models and replays physics state frame-by-frame.

### RL Environment for Sound Localization (Phase 4A)
```python
from physonic.envs import SoundLocalizationEnv

# Create environment
env = SoundLocalizationEnv(
    audio_duration=1.0,              # 1 second audio clips
    use_mel_spectrogram=True,        # Use mel-spectrogram (vs raw audio)
    n_mels=64,                       # Number of mel bands
    material="steel",                # Impact material
    scenario_type="drop"             # Physics scenario
)

# Single episode
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")  # (64, 87, 2) - mel-spectrogram
print(f"True azimuth: {info['true_azimuth']:.1f}°")

# Agent predicts azimuth
predicted_azimuth = 45.0  # Example prediction
obs, reward, terminated, truncated, info = env.step([predicted_azimuth])
print(f"Reward: {reward:.3f}")  # Range: [-1, 0]
print(f"Angular error: {info['angular_error']:.1f}°")

# Training loop with Stable-Baselines3
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

### RL Environment for Sound Classification (Phase 4B)
```python
from physonic.envs import SoundClassificationEnv
import numpy as np

# Create environment
env = SoundClassificationEnv(
    audio_duration=0.5,               # 0.5s audio for faster testing
    use_mel_spectrogram=True,         # Use mel-spectrogram features
    randomize_position=True,          # Vary source distance/azimuth
    randomize_surface=False           # Fixed surface (easier task)
)

# Single episode
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")     # (64, 44, 2) - mel-spectrogram
print(f"True material: {info['true_material']}")  # steel, glass, wood, or rubber
print(f"Action space: {env.action_space}")   # Discrete(4)

# Agent predicts material class
predicted_class = 1  # Guess: glass
obs, reward, terminated, truncated, info = env.step(predicted_class)
print(f"Prediction: {info['predicted_material']}")
print(f"Correct: {info['correct']}")
print(f"Reward: {reward}")  # 1.0 if correct, 0.0 if wrong

# Run multiple episodes and track confusion matrix
for _ in range(100):
    obs, info = env.reset()
    action = env.action_space.sample()  # Random guess
    env.step(action)

# View confusion matrix
cm = env.get_confusion_matrix()
print("\nConfusion Matrix:")
print(cm)

# Per-class accuracy
for i, material in enumerate(env.MATERIAL_CLASSES):
    if cm[i].sum() > 0:
        acc = cm[i, i] / cm[i].sum()
        print(f"{material}: {100*acc:.1f}%")
```

### RL Environment for Selective Response (Phase 4C)
```python
from physonic.envs import SelectiveResponseEnv

# Create environment
env = SelectiveResponseEnv(
    target_materials=["glass"],        # Materials that require response
    audio_duration=0.5,                # 0.5s audio for faster testing
    target_probability=0.3,            # 30% targets, 70% distractors
    reward_tp=1.0,                     # True positive reward
    reward_tn=0.5,                     # True negative reward
    reward_fp=-1.0,                    # False positive penalty
    reward_fn=-2.0                     # False negative penalty (worse)
)

# Single episode
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")        # (64, 44, 2) - mel-spectrogram
print(f"Is target: {info['is_target']}")        # True if target, False if distractor
print(f"Material: {info['material']}")          # steel, glass, wood, or rubber
print(f"Action space: {env.action_space}")      # Discrete(2): 0=IGNORE, 1=RESPOND

# Agent decides whether to respond
action = 1 if should_respond else 0  # Binary decision
obs, reward, terminated, truncated, info = env.step(action)
print(f"Outcome: {info['outcome']}")  # TP, TN, FP, or FN
print(f"Reward: {reward}")            # Varies by outcome

# Run multiple episodes and evaluate precision/recall
for _ in range(100):
    obs, info = env.reset()
    # Perfect agent strategy
    action = 1 if info['is_target'] else 0
    env.step(action)

# Get precision/recall/F1 metrics
metrics = env.get_metrics()
print(f"\nMetrics:")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall:    {metrics['recall']:.3f}")
print(f"F1 Score:  {metrics['f1']:.3f}")
print(f"Accuracy:  {metrics['accuracy']:.3f}")

# View 2x2 confusion matrix
cm = env.get_confusion_matrix()
print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
print(cm)

# PyTorch training example (optional)
# See scripts/train_selective_response.py for REINFORCE implementation
```

**Organized directory structure**:
```
outputs/
├── mono/              # Mono audio (8 scenarios)
│   ├── {scenario}.wav
│   ├── {scenario}_telemetry.npz
│   ├── {scenario}_events.json
│   └── videos/        # (optional) rendered videos
├── spatial/           # Binaural spatial audio
├── reverb/            # Spatial + room acoustics
├── multi/             # Multi-object scenarios
└── custom/            # Custom experiments

datasets/
└── physonic_v1/
    ├── data/chunk-000.parquet      # State observations at fps (30 Hz default)
    ├── audio/binaural/*.flac       # Audio chunks (1470 samples @ 30 fps)
    ├── videos/                     # (optional) synchronized MP4 videos
    │   ├── episode_00000_front.mp4
    │   ├── episode_00000_side.mp4
    │   └── episode_00000_top.mp4
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

6. **Video rendering telemetry-based**: Post-hoc rendering from saved telemetry (not real-time during simulation). Allows flexible camera placement after simulation completes.

## Code Organization

**Entry points**:
- `run_demo.py`: Single-object scenarios with spatial + room acoustics
- `run_demo_multi.py`: Multi-object scenarios (7 predefined scenarios)
- `export_dataset.py`: Convert outputs to LeRobotDataset format (with optional video rendering)
- `render_video.py`: Standalone post-hoc video rendering from saved telemetry
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
- `export/writer.py`: LeRobotWriter, audio chunking, telemetry downsampling, optional video rendering
- `export/schemas.py`: PyArrow schema definitions for Parquet (including video path columns)
- `export_dataset.py`: CLI tool for batch export with optional video rendering

**Video rendering**:
- `render/camera.py`: CameraConfig, CameraPreset, camera mode helpers
- `render/renderer.py`: VideoRenderer with MuJoCo offscreen rendering, telemetry replay
- `render/encoder.py`: VideoEncoder with H.264 MP4 encoding, automatic codec fallback
- `render_video.py`: Standalone CLI tool for post-hoc video rendering

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

### ✅ Phase 3B: Multi-Camera Video Rendering (Complete)
**Weeks 1-2: Core rendering infrastructure**
- Camera configuration system (CameraConfig, CameraPreset)
- 4 camera modes: fixed (front/side/top), multi-camera, tracking, egocentric
- MuJoCo offscreen rendering with mujoco.Renderer
- Post-hoc rendering from telemetry replay
- MP4 encoding with H.264 (automatic codec fallback)
- Standalone CLI tool: `python3 -m physonic.render_video`
- Spec reconstruction from episode names (reconstruct_scene_spec, reconstruct_multi_scene_spec)
- 73 tests passing (camera, encoder, renderer, reconstruction)

**Week 3: Dataset export integration**
- Video rendering integrated into LeRobotWriter
- Parquet schema updated with video path columns (observation.video.front/side/top)
- CLI args: `--include-video`, `--video-cameras`, `--video-width`, `--video-height`
- Output: `videos/episode_{N:05d}_{camera}.mp4`
- 5 additional integration tests for video rendering
- End-to-end verification: simulation → video rendering → dataset export

**Week 4: Validation & finalization**
- End-to-end pipeline tested: 8 episodes, 600 frames, synchronized audio+video
- Videos: H.264 @ 30fps, synchronized with FLAC audio chunks
- All 185 tests passing (181 passed, 4 skipped)
- Documentation updated

### ✅ Phase 3C: Environment Templates with Procedural Generation (Partial Complete)
**Sound Synthesis Infrastructure** (Complete):
- `physonic/synthesis/continuous.py`: Continuous equipment sounds via additive synthesis
  - 4 presets: server_hum (120Hz), fan (80Hz), cooling_system (180Hz), hvac (60Hz)
  - Harmonics + filtered noise + amplitude modulation
- `physonic/synthesis/alerts.py`: Alert/beep sounds via pure tones + ADSR envelopes
  - 6 presets: beep, double_beep, alarm, warning, chime, error
  - Timeline placement utilities for triggered events
- `physonic/synthesis/mixer.py`: Multi-source acoustic scene mixing
  - Combines impact + continuous + alert sounds
  - Automatic normalization and soft limiting
  - Data center ambient sound generator
- 72 unit tests passing (all synthesis functionality)

**Layout Generation** (Complete):
- `physonic/environments/layout.py`: Grid-based procedural layout engine
  - Collision detection with placement constraints
  - Row/grid pattern generation algorithms
  - Occupancy tracking and density limits
- `physonic/environments/objects.py`: Static object library for MuJoCo
  - 13 object types (server racks, pallets, shelves, refrigerators, etc.)
  - MuJoCo XML generation with mocap bodies (non-moving, collision-enabled)
  - Equipment sound integration

**Data Center Template** (Complete):
- `physonic/environments/data_center.py`: Procedural data center generation
  - 3 layout types: hot_aisle (parallel rows), pod (clustered), perimeter (wall-mounted)
  - 3 scenarios: maintenance (tool sounds + beep), failure (alarms), normal (ambient only)
  - Generates: static objects + continuous sounds + alert events
  - Example: 8 racks + 2 cooling units + 7 equipment sounds + scenario-specific alerts

**Remaining** (Deferred):
- Warehouse environment template (shelving, forklifts, package drops)
- Supermarket environment template (carts, checkout, refrigerator hum)
- CLI integration for environment generation

### Phase 4: RL Environments for Acoustic Perception
**4A: Sound Source Localization** ✅ COMPLETED
- ✅ Gymnasium environment with audio observation space (mel-spectrograms or raw audio)
- ✅ Continuous azimuth action space [-180°, 180°]
- ✅ Angular error reward: `-|pred - true| / 180`
- ✅ Integration with physics simulation and modal synthesis
- ✅ Seed reproducibility and Gymnasium API compliance
- ✅ Comprehensive test suite (41 tests passing)
- Module: `physonic/envs/` with `SoundLocalizationEnv`, preprocessing, rewards

**4B: Sound Classification** ✅ COMPLETED
- ✅ Gymnasium environment for material classification (steel, glass, wood, rubber)
- ✅ Discrete action space (4 classes)
- ✅ Binary reward: +1.0 correct, 0.0 incorrect
- ✅ Confusion matrix tracking built-in
- ✅ Position/surface randomization options
- ✅ CPU-friendly design (no pre-trained models required)
- ✅ Comprehensive test suite (21 tests passing)
- ✅ Baseline agents: random (~25% accuracy), heuristic (~27% accuracy)
- Module: `physonic/envs/classification.py` with `SoundClassificationEnv`

**4C: Selective Response** ✅ COMPLETED
- ✅ Gymnasium environment for binary decision-making (respond/ignore)
- ✅ Configurable target materials and distractor materials
- ✅ Asymmetric reward structure (TP/TN/FP/FN penalties)
- ✅ Precision/Recall/F1/Accuracy metrics built-in
- ✅ 2x2 confusion matrix tracking
- ✅ Target probability control for class imbalance
- ✅ Comprehensive test suite (27 tests passing)
- ✅ Baseline agents: random, always-respond, never-respond
- ✅ PyTorch training example with REINFORCE
- Module: `physonic/envs/selective_response.py` with `SelectiveResponseEnv`

**4D: Multi-Modal Fusion**
- Vision + audio observation space
- Robust perception with sensor dropout
- Attention mechanism analysis

### ✅ Demo and Visualization Tools (Complete)
**Episode Recording Infrastructure**:
- `physonic/visualization/recorder.py`: Episode recording compatible with Gymnasium VideoRecorder API
  - EpisodeRecorder class for saving/loading RL episodes
  - Saves: observations (NPY), actions, rewards, infos (JSON), audio (WAV)
  - Export stubs for HuggingFace datasets and Isaac Sim USD format (future)
  - JSON metadata with environment info and episode statistics

**Visualization Utilities**:
- `physonic/visualization/plots.py`: Comprehensive plotting functions
  - `plot_waveform()`: Audio waveform visualization (mono/stereo)
  - `plot_spectrogram()`: STFT spectrogram with dB scale
  - `plot_mel_spectrogram()`: Mel-spectrogram from RL observations
  - `plot_metrics()`: Episode rewards with rolling average
  - `plot_confusion_matrix()`: Normalized confusion matrix with percentages
  - `plot_episode_summary()`: Multi-panel overview (observation + rewards + actions + metrics)

**Interactive Demo Scripts**:
- `demo_rl_env.py`: Interactive demo for all three RL environments
  - Supports localization, classification, selective response
  - Random agent and perfect agent (ground truth cheating)
  - Automatic episode recording and visualization generation
  - Usage: `python3 demo_rl_env.py --env localization --episodes 10 --visualize`

- `demo_environment_gen.py`: Interactive demo for procedural environment generation
  - Data center scenarios (maintenance, failure, normal)
  - Layout types (hot_aisle, pod, perimeter)
  - Outputs: MuJoCo XML, acoustic audio, layout visualization, JSON summary
  - Usage: `python3 demo_environment_gen.py --scenario maintenance --layout hot_aisle --visualize`

- `view_environment.py`: 3D environment viewer
  - Multi-angle static rendering (Front, Top, Side, 3D View)
  - Interactive real-time viewer support (requires mjpython on macOS)
  - View existing XML scenes or generate environments on-the-fly
  - MuJoCo offscreen rendering at 480x480 resolution
  - Usage: `python3 view_environment.py --xml scene.xml` or `python3 view_environment.py --scenario maintenance --layout hot_aisle`

**Compatibility**:
- ✅ Gymnasium API (VideoRecorder pattern)
- 🔄 HuggingFace datasets (export stub ready)
- 🔄 Isaac Sim USD (export stub ready for future integration)

### ✅ Recent Updates (2026-10-13)

**Project Organization** (Complete):
- ✅ Reorganized all Python scripts from root into `scripts/` subdirectories
- ✅ Created organized structure: baselines/, demos/, training/, visualization/, utils/, examples/
- ✅ Updated README.md with new script paths and Example 11 (PPO training)
- ✅ Created scripts/README.md with comprehensive organization policy
- ✅ Verified: No Python files at project root (clean structure)

**PPO Training Infrastructure** (Complete):
- ✅ `scripts/training/train_ppo_data_center.py`: Full PPO training with video recording
  - 3 types of video recording: evaluation videos, training rollouts, real-time viewer
  - Supports both parallel (SubprocVecEnv) and single (DummyVecEnv) environments
  - Automatic warnings when features require n-envs=1
  - TensorBoard logging and checkpoint saving
- ✅ `scripts/training/video_recorder_callback.py`: Evaluation episode videos
  - Records deterministic policy performance at regular intervals
  - 480×480 resolution @ 10fps with H.264 encoding
  - Overlays: step count, reward, distance to goal
- ✅ `scripts/training/training_monitor_callback.py`: Training rollout videos
  - Records actual training (stochastic policy with exploration)
  - Shows what the agent is actually learning during training
  - Optional real-time viewer (very slow, debugging only)

**Video Recording Codec Fix** (Complete):
- ✅ **Problem**: Videos not being created on macOS despite success messages
- ✅ **Root Cause**: OpenCV built with AVFoundation (not FFmpeg), 'mp4v' codec incompatible
- ✅ **Solution**: Changed codec from 'mp4v' to 'avc1' (H.264) in both callbacks
- ✅ **Verification**: Created test diagnostic tool (`test_video_recording.py`)
- ✅ **Result**: Videos now created successfully (verified with ffprobe)
- ✅ **Documentation**: Full details in `VIDEO_RECORDING_CODEC_FIX.md`

**Audio Perception Visualization** (Complete):
- ✅ `scripts/visualization/visualize_audio_observation.py`: Shows how agents perceive audio
  - Compares raw waveform (44,100 samples) vs mel-spectrogram (6,400 features)
  - Visualizes binaural encoding (left/right channels for ITD/ILD)
  - Generates PPO architecture diagram (6,406 input features → 256 → 2 actions)
  - Outputs: `audio_perception_visualization.png`, `ppo_architecture_diagram.png`
- ✅ `AUDIO_AND_VIDEO_GUIDE.md`: Comprehensive guide explaining:
  - How agents process audio (mel-spectrograms as 2D "photographs of sound")
  - Three video recording options (evaluation, training, real-time)
  - Performance impact analysis

**Training Configuration**:
```bash
# Full training with all features (requires n-envs=1 for training videos)
PYTHONPATH=/Users/macbook/Desktop/rightsify/physonic_mvp \
python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 100000 \
  --n-envs 1 \
  --record-video \
  --monitor-training \
  --video-freq 10000 \
  --monitor-freq 5000

# Parallel training with evaluation videos only (faster)
PYTHONPATH=/Users/macbook/Desktop/rightsify/physonic_mvp \
python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 100000 \
  --n-envs 4 \
  --record-video \
  --video-freq 10000
```

**Output Directories**:
- `outputs/ppo_datacenter/models/`: Saved models (checkpoints + best + final)
- `outputs/ppo_datacenter/logs/`: TensorBoard logs
- `outputs/ppo_datacenter/eval_videos/`: Evaluation episode videos (deterministic)
- `outputs/ppo_datacenter/training_videos/`: Training rollout videos (stochastic)

### Technical Debt to Address
- Fine-tune Schroeder RT60 decay calibration
- Replace material approximations with mesh-derived modal data
- Validate synthesis against real-world recordings (perceptual studies)
- VBAP or spherical harmonics for smoother HRTF interpolation
- USD/Omniverse export for NVIDIA Isaac Sim integration
- WebSocket/gRPC streaming for online RL training
- Implement real audio rendering in data center navigation environment (currently using placeholder zeros)
