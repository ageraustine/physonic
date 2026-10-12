# Physonic: Physics-Driven Acoustic Synthesis for Robot Perception

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.0+-orange.svg)](https://mujoco.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Physonic** generates realistic acoustic training data for robot perception systems by simulating physics-based contact sounds. It combines MuJoCo rigid-body dynamics with modal synthesis to produce synchronized audio, physics state, and ground truth labels—ideal for training robots to understand their environment through sound.

https://github.com/user-attachments/assets/example-video-placeholder.mp4

*Example: Multi-object collision with binaural spatial audio and room acoustics*

## Why Physonic?

Modern robots need to perceive their environment through **multiple modalities**. While vision is well-studied, **acoustic perception** remains underexplored due to lack of diverse, labeled training data. Physonic addresses this by:

- **Physically grounded synthesis**: Contact sounds derived from real material properties (compliance, damping, resonant modes)
- **Scalable data generation**: Procedurally generate thousands of scenarios with controlled variation
- **Multi-modal output**: Synchronized audio, physics state, video (coming soon), and ground truth labels
- **Spatial realism**: HRTF-based binaural audio and room acoustics for warehouse, office, and industrial environments
- **Robot-ready format**: Export to LeRobotDataset (Parquet) for seamless integration with Hugging Face ecosystem

**Use cases**:
- **Sound source localization** in 3D space (RL environment included)
- **Material classification** from impact sounds (RL environment included)
- **Selective response** to task-relevant sounds (RL environment included)
- Event detection (collisions, sliding, rolling)
- Reinforcement learning for contact-rich manipulation
- Acoustic anomaly detection (identifying mechanical failures)

---

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Physics Simulation** | ✅ | MuJoCo 5kHz contact dynamics (single + multi-object) |
| **Acoustic Synthesis** | ✅ | Modal synthesis (impact + friction sounds) |
| **Spatial Audio** | ✅ | HRTF-based binaural rendering (MIT KEMAR) |
| **Room Acoustics** | ✅ | Early reflections + late reverb (17 room templates) |
| **Dataset Export** | ✅ | LeRobotDataset format (Parquet + FLAC + MP4) |
| **Video Rendering** | ✅ | Multi-camera RGB (4 camera modes, H.264 encoding) |
| **RL Environments** | ✅ | Gymnasium integration (3 acoustic perception tasks) |

---

## Installation

### Prerequisites
- **Python 3.9+**
- **MuJoCo 3.0+** ([installation guide](https://mujoco.readthedocs.io/en/stable/programming/index.html#getting-started))

### Quick Install
```bash
# Clone repository
git clone https://github.com/ageraustine/physonic.git
cd physonic

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download HRTF dataset (optional, for spatial audio)
# MIT KEMAR HRTF dataset should be placed in assets/hrtf/test_kemar/
# See: https://sound.media.mit.edu/resources/KEMAR.html
```

### Verify Installation
```bash
# Run quick test
python3 -m physonic.run_demo --out outputs/test --seed 42

# Should generate 8 audio files
ls outputs/test/*.wav
```

---

## Quick Start

All outputs are organized in the `outputs/` directory with subdirectories for different experiment types.

### 1. Generate Single-Object Scenarios (Mono Audio)
```bash
python3 -m physonic.run_demo --out outputs/mono --seed 0
```

**Output**: `outputs/mono/` with 8 scenarios (4 drop + 4 slide) × (steel, glass, wood, rubber surfaces)
- `drop_steel_on_glass.wav` - Sphere dropped from 0.5m
- `slide_wood_on_rubber.wav` - Box sliding at 3 m/s
- `*_telemetry.npz` - Contact forces, positions, velocities (5kHz)
- `*_events.json` - Discrete impact events with times, impulses

### 2. Add Spatial Audio (Binaural)
```bash
python3 -m physonic.run_demo --spatial --hrtf-path assets/hrtf/test_kemar --out outputs/spatial
```

**Output**: `outputs/spatial/` with binaural audio
- Left/right ear signals encode 3D sound source position
- Accurate interaural time difference (ITD) and level difference (ILD)

### 3. Add Room Acoustics
```bash
# Use predefined room template
python3 -m physonic.run_demo --spatial --room warehouse_large --out outputs/reverb

# Or customize room parameters
python3 -m physonic.run_demo --spatial \
  --room-dim 30,20,8 \
  --rt60 2.5 \
  --absorption 0.15 \
  --out outputs/custom
```

**Available room templates**: `warehouse_large`, `data_center`, `supermarket_medium`, `office_small`, `hallway_narrow`, `factory_floor`, `anechoic` ([see all 17 templates](physonic/acoustics/rooms.py))

### 4. Multi-Object Scenarios
```bash
# Available: two_sphere_collision, stack_collapse, sphere_cascade,
#            tumble_box, scatter_pile, domino_chain, rolling_cylinders

python3 -m physonic.run_demo_multi --scenario stack_collapse --spatial --out outputs/multi
```

**Output**: `outputs/multi/` with complex acoustic scenes
- Per-pair contact tracking
- Object-object collision sounds
- Ground truth labels for each object (material, position, velocity)

### 5. Export to LeRobotDataset
```bash
# Export all scenarios from outputs/mono to Parquet dataset
python3 -m physonic.export_dataset \
  --input outputs/mono \
  --output datasets/physonic_v1 \
  --fps 30
```

**Output structure**:
```
outputs/
├── mono/              # Mono audio (8 scenarios)
├── spatial/           # Binaural spatial audio
├── reverb/            # Spatial + room acoustics
├── multi/             # Multi-object scenarios
└── custom/            # Custom experiments

datasets/
└── physonic_v1/
    ├── data/chunk-000.parquet
    ├── audio/binaural/*.flac
    └── meta/info.json
```

**Parquet schema**:
| Column | Type | Description |
|--------|------|-------------|
| `observation.state` | list\[float32\] | Physics state vector (4 or 6N+2 floats) |
| `observation.audio.binaural` | string | Path to FLAC audio chunk |
| `ground_truth.events` | string | JSON list of events in this frame |
| `ground_truth.materials` | string | JSON dict of object materials |
| `ground_truth.sound_sources` | string | JSON list of sound source metadata |
| `episode_index` | int64 | Episode ID |
| `frame_index` | int64 | Frame number within episode |
| `timestamp` | float64 | Time from episode start (seconds) |

---

## Dataset Format Details

### State Vector
**Single-object scenarios** (4 floats):
```python
[z, vz, fn, vt]
# z:  object height (m)
# vz: vertical velocity (m/s)
# fn: normal contact force (N)
# vt: tangential sliding speed (m/s)
```

**Multi-object scenarios** (6N+2 floats for N objects):
```python
[obj1_x, obj1_y, obj1_z, obj1_vx, obj1_vy, obj1_vz,
 obj2_x, obj2_y, obj2_z, obj2_vx, obj2_vy, obj2_vz,
 ...,
 total_fn, total_ft]
# Per object: 3D position + 3D velocity
# Aggregates: total normal force, total tangential force
```

### Audio Chunking
- **Sample rate**: 44.1 kHz
- **Frame rate**: 30 fps (configurable with `--fps`)
- **Samples per frame**: 1470 @ 30 fps
- **Format**: FLAC (lossless compression)

### Timestamp Synchronization
Three timelines are synchronized:
- **Physics**: 5 kHz (0.0002s timestep)
- **Audio**: 44.1 kHz (continuous)
- **Parquet frames**: 30 fps (discrete observations)

Parquet frame times are the source of truth. Audio is chunked into frame-aligned segments, and physics telemetry is downsampled using nearest-neighbor interpolation.

---

## Examples

### Example 1: Material Classification Dataset
Generate 1000 drop sounds across 4 materials with randomized heights:

```python
import numpy as np
from physonic.contact.scene import SceneSpec
from physonic.contact.rig import simulate_scene
from physonic.contact.modal_synth import render_audio

materials = ['steel', 'glass', 'wood', 'rubber']
heights = np.random.uniform(0.3, 1.0, size=1000)

for i, (mat, h) in enumerate(zip(np.random.choice(materials, 1000), heights)):
    spec = SceneSpec(kind='drop', surface='steel', obj=mat, drop_height=h)
    tele, events = simulate_scene(spec)
    audio = render_audio(tele, events, spec)
    # Save audio, tele, events...
```

### Example 2: Sound Source Localization Dataset
Generate binaural audio from random 3D positions:

```python
from physonic.spatial.renderer import render_spatial_scene

azimuths = np.random.uniform(-180, 180, size=500)
elevations = np.random.uniform(-40, 90, size=500)
distances = np.random.uniform(0.5, 3.0, size=500)

for i, (az, el, dist) in enumerate(zip(azimuths, elevations, distances)):
    source_pos = spherical_to_cartesian(az, el, dist)
    binaural_audio = render_spatial_scene(
        spec,
        source_pos=source_pos,
        listener_pos=[0, 0, 1.5],
        hrtf_path='assets/hrtf/test_kemar'
    )
    # Ground truth: az, el, dist
```

### Example 3: RL Environment for Sound Localization
```python
from physonic.envs import SoundLocalizationEnv

# Create environment
env = SoundLocalizationEnv(
    audio_duration=1.0,              # 1 second audio clips
    use_mel_spectrogram=True,        # Use mel-spectrogram (vs raw audio)
    n_mels=64,                       # Number of mel bands
    material="steel",                # Impact material
    hrtf_path="assets/hrtf/test_kemar/"
)

# Training loop
for episode in range(100):
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")  # (64, 87, 2) - mel-spectrogram
    print(f"True azimuth: {info['true_azimuth']:.1f}°")

    # Agent predicts azimuth
    predicted_azimuth = 45.0  # Example prediction
    obs, reward, terminated, truncated, info = env.step([predicted_azimuth])
    print(f"Reward: {reward:.3f}")  # Range: [-1, 0]
    print(f"Angular error: {info['angular_error']:.1f}°")
```

### Example 4: RL Environment for Material Classification
```python
from physonic.envs import SoundClassificationEnv

# Create environment
env = SoundClassificationEnv(
    audio_duration=0.5,               # 0.5s audio for faster testing
    use_mel_spectrogram=True,
    randomize_position=True,          # Vary source distance/azimuth
    hrtf_path="assets/hrtf/test_kemar/"
)

# Training loop with confusion matrix
for episode in range(100):
    obs, info = env.reset()
    print(f"True material: {info['true_material']}")  # steel, glass, wood, or rubber

    # Agent predicts material class (0=steel, 1=glass, 2=wood, 3=rubber)
    predicted_class = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(predicted_class)
    print(f"Correct: {info['correct']}, Reward: {reward}")  # 1.0 if correct, 0.0 if wrong

# View confusion matrix
cm = env.get_confusion_matrix()
print("\nConfusion Matrix (rows=true, cols=predicted):")
print(cm)
```

### Example 5: RL Environment for Selective Response
```python
from physonic.envs import SelectiveResponseEnv

# Create environment
env = SelectiveResponseEnv(
    target_materials=["glass"],        # Materials that require response
    audio_duration=0.5,
    target_probability=0.3,            # 30% targets, 70% distractors
    reward_tp=1.0,                     # True positive reward
    reward_tn=0.5,                     # True negative reward
    reward_fp=-1.0,                    # False positive penalty
    reward_fn=-2.0,                    # False negative penalty (worse)
    hrtf_path="assets/hrtf/test_kemar/"
)

# Training loop
for episode in range(100):
    obs, info = env.reset()
    print(f"Is target: {info['is_target']}, Material: {info['material']}")

    # Agent decides: 0=IGNORE, 1=RESPOND
    action = 1 if should_respond else 0
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Outcome: {info['outcome']}, Reward: {reward}")  # TP, TN, FP, or FN

# Get precision/recall/F1 metrics
metrics = env.get_metrics()
print(f"\nPrecision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1']:.3f}")
```

### Example 6: Loading Dataset in Python
```python
import pandas as pd
import soundfile as sf

# Load Parquet
df = pd.read_parquet('datasets/physonic_v1/data/chunk-000.parquet')

# Access first frame
row = df.iloc[0]
state = row['observation.state']  # [z, vz, fn, vt]
audio_path = row['observation.audio.binaural']  # 'audio/binaural/chunk-00000.flac'

# Load corresponding audio
audio, sr = sf.read(f'datasets/physonic_v1/{audio_path}')
print(f"State: {state}, Audio shape: {audio.shape}, Sample rate: {sr}")
```

### Example 7: Hugging Face Datasets Integration
```python
from datasets import load_dataset

# Load from local directory
dataset = load_dataset('parquet', data_dir='datasets/physonic_v1/data')

# Or push to Hugging Face Hub
dataset.push_to_hub('ageraustine/physonic-warehouse-v1')

# Later, load from Hub
dataset = load_dataset('ageraustine/physonic-warehouse-v1')
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MuJoCo Physics Engine                    │
│  • Rigid-body dynamics (5 kHz timestep)                     │
│  • Contact force telemetry (normal, tangential, position)   │
│  • Multi-object collision detection                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Event Extraction                          │
│  • Detect impact onsets from force spikes                   │
│  • Compute impulse (integral of excess force)               │
│  • Filter micro-hops (J/mass < 0.3 m/s threshold)           │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  Modal Synthesis                            │
│  • Impacts: Decaying sinusoids (resonant modes)             │
│    - Material → frequencies, damping, amplitudes            │
│    - Impulse → mode excitation levels                       │
│  • Friction: Broadband noise + modal ringing                │
│    - Envelope: sqrt(Fn) × sliding_speed                     │
│  Output: Mono audio @ 44.1 kHz                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Room Acoustics (Optional)                      │
│  • Early reflections: Image-source method (order 2-3)       │
│  • Late reverb: Schroeder-Moorer (4 comb + 2 allpass)       │
│  • RT60-based gain tuning                                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│             Spatial Rendering (Optional)                    │
│  • Distance attenuation (1/r law)                           │
│  • HRTF binaural filtering (MIT KEMAR)                      │
│    - 5° resolution, 128-sample impulses                     │
│    - Encodes ITD (±0.65ms) and ILD                          │
│  Output: Binaural stereo @ 44.1 kHz                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 Dataset Export                              │
│  • Audio chunking (1470 samples/frame @ 30 fps)             │
│  • Telemetry downsampling (5 kHz → 30 fps)                  │
│  • State vector construction                                │
│  • Parquet + FLAC export (LeRobotDataset format)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Material Properties

Physonic includes a database of 4 acoustic materials with validated properties:

| Material | f₀ (Hz) | t₆₀ (s) | Compliance | Use Case |
|----------|---------|---------|------------|----------|
| **Steel** | 800 | 0.8 | Low (0.01) | Rigid, metallic impacts (sharp, ringing) |
| **Glass** | 1200 | 0.5 | Low (0.02) | Brittle, high-frequency (crisp, short decay) |
| **Wood** | 400 | 0.3 | Medium (0.1) | Natural, warm tones (dull thud) |
| **Rubber** | 150 | 0.15 | High (0.5) | Soft, damped (muted, quick decay) |

Each material defines:
- **Fundamental frequency** (f₀) and **mode ratios** (harmonic structure)
- **Decay time** (t₆₀) and **damping slope** (frequency-dependent damping)
- **Compliance** (Hertz contact time → affects impact spectrum shape)

---

## Advanced Usage

### Custom Materials
```python
from physonic.contact.materials import Material

# Define custom material
carbon_fiber = Material(
    name='carbon_fiber',
    density=1600,          # kg/m³
    youngs=150e9,          # Pa
    poissons=0.3,
    compliance=0.015,
    f0=1500,               # Fundamental frequency (Hz)
    mode_ratios=[1.0, 2.7, 5.4, 8.9],  # Harmonic structure
    t60=0.6,               # Decay time (seconds)
    damp_slope=0.0002      # Frequency-dependent damping
)

# Use in scene
spec = SceneSpec(kind='drop', surface='steel', obj=carbon_fiber)
```

### Procedural Scenario Generation
```python
from physonic.contact.scenarios import MultiSceneSpec

# Generate random stack of objects
spec = MultiSceneSpec(
    kind='stack_collapse',
    n_objects=5,
    materials=['steel', 'wood', 'glass'],  # Random selection
    masses=np.random.uniform(0.5, 2.0, size=5),
    sizes=np.random.uniform(0.05, 0.15, size=5),
    initial_height=1.5
)
```

### Batch Export with Custom Preprocessing
```python
from physonic.export.writer import LeRobotWriter
from physonic.export.episode import load_episode

writer = LeRobotWriter(
    output_dir='datasets/physonic_custom',
    fps=60,  # Higher frame rate
    chunk_size=500
)

for episode_dir in episode_dirs:
    episode = load_episode(episode_dir)

    # Custom preprocessing
    episode.audio = apply_noise(episode.audio, snr_db=30)
    episode.audio = apply_eq(episode.audio, freq=200, gain=-3)

    writer.add_episode(episode)

writer.write()
writer.finalize()
```

---

## Testing

```bash
# Run all tests
pytest -v

# Run specific test modules
pytest tests/test_contact/ -v           # Physics & synthesis (19 tests)
pytest tests/test_spatial/ -v           # Spatial audio (44 tests)
pytest tests/test_acoustics/ -v         # Room acoustics (28 tests)
pytest tests/test_export/ -v            # Dataset export (21 tests)
pytest tests/test_render/ -v            # Video rendering (73 tests)
pytest tests/test_envs/ -v              # RL environments (89 tests)

# Run integration tests only
pytest -m integration -v
```

**Test coverage**: 274 tests covering physics simulation, acoustic synthesis, spatial audio, room acoustics, video rendering, dataset export, and RL environments (sound localization, material classification, selective response).

---

## Performance

Benchmarks on MacBook Pro (M1, 16GB RAM):

| Operation | Time | Notes |
|-----------|------|-------|
| Single drop scenario | ~0.8s | 2.5s simulation @ 5kHz |
| Multi-object (5 objects) | ~3.2s | Stack collapse scenario |
| Binaural HRTF convolution | +0.3s | Per scenario |
| Room acoustics | +0.2s | Image-source + reverb |
| Dataset export (100 episodes) | ~45s | @ 30 fps, FLAC compression |

**Scalability**: Parallel generation using `multiprocessing` can process 500+ scenarios in ~10 minutes.

---

## Roadmap

### ✅ Phase 3B: Multi-Camera Video Rendering (Complete)
- [x] MuJoCo offscreen RGB rendering
- [x] Multi-camera support (front, side, top views)
- [x] Tracking camera mode (follows specific object)
- [x] Egocentric camera mode (attached to object)
- [x] MP4 encoding with H.264 codec
- [x] Dataset integration (synchronized video + audio)

### ✅ Phase 4A-4C: RL Environments (Complete)
- [x] Gymnasium integration with audio observation space
- [x] Sound localization task (azimuth prediction)
- [x] Material classification task (4-class discrete action)
- [x] Selective response task (binary respond/ignore decisions)
- [x] Precision/Recall/F1 metrics for evaluation
- [x] Baseline agents and PyTorch training examples

### Phase 3C: Environment Templates
- [ ] Warehouse: forklifts, pallets, conveyor belts
- [ ] Data center: server racks, cooling systems
- [ ] Supermarket: shopping carts, shelving units
- [ ] Procedural layout generation

### Phase 4D: Multi-Modal Fusion
- [ ] Vision + audio observation spaces
- [ ] Sensor dropout and multi-modal robustness
- [ ] Attention mechanism for modality fusion
- [ ] Contact-rich manipulation tasks

### Future Enhancements
- [ ] Mesh-derived modal analysis (replace analytical modes)
- [ ] Perceptual validation studies
- [ ] VBAP/ambisonics for smoother spatial interpolation
- [ ] Real-time synthesis for online RL training
- [ ] USD/Omniverse export for NVIDIA Isaac Sim

---

## Citation

If you use Physonic in your research, please cite:

```bibtex
@software{physonic2025,
  title = {Physonic: Physics-Driven Acoustic Synthesis for Robot Perception},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/ageraustine/physonic},
  note = {Version 1.0}
}
```

---

## Contributing

Contributions are welcome! Areas of interest:
- **Materials**: Add more acoustic materials (metals, ceramics, fabrics)
- **Scenarios**: Design new multi-object interactions
- **Validation**: Perceptual studies comparing synthetic vs. real recordings
- **Environments**: Implement procedural scene generation
- **RL Tasks**: Design acoustic perception benchmarks

Please open an issue or pull request on [GitHub](https://github.com/ageraustine/physonic).

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **MuJoCo**: Physics simulation ([mujoco.org](https://mujoco.org))
- **MIT KEMAR**: HRTF dataset ([sound.media.mit.edu](https://sound.media.mit.edu/resources/KEMAR.html))
- **LeRobot**: Dataset format specification ([huggingface.co/lerobot](https://huggingface.co/lerobot))
- Modal synthesis inspired by James et al. (2006) - "Precomputed Acoustic Transfer"

---

## Support

- **Documentation**: See [CLAUDE.md](CLAUDE.md) for detailed architecture and development guide
- **Issues**: [GitHub Issues](https://github.com/ageraustine/physonic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ageraustine/physonic/discussions)

---

**Built for the robot perception community** 🤖🔊
