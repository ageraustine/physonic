# Audio Perception and Video Rendering Guide

This guide explains how RL agents perceive audio and how to visualize training with video rendering.

## Table of Contents
1. [How Agents Perceive Audio](#how-agents-perceive-audio)
2. [Video Rendering During Training](#video-rendering-during-training)
3. [Examples](#examples)

---

## How Agents Perceive Audio

### Overview

Robots/agents don't "hear" audio the same way humans do. They receive audio as numerical data that must be processed by neural networks. Physonic supports two observation formats:

### 1. Raw Audio Waveform

**Format**: Time-series signal
**Shape**: `(samples, channels)`
**Example**: `(22050, 2)` = 0.5 seconds @ 44.1kHz, stereo

```
Raw audio is the actual sound wave:
     Amplitude
        │
    1.0 │     ╱╲    ╱╲
        │    ╱  ╲  ╱  ╲
    0.0 │───────────────────  Time
        │  ╱    ╲╱    ╲
   -1.0 │╱              ╲
```

**Advantages**:
- ✓ Complete frequency information
- ✓ No information loss

**Disadvantages**:
- ✗ Too much data for deep learning (22,050 features per 0.5s)
- ✗ Hard for neural networks to learn patterns
- ✗ Requires very deep networks (like WaveNet)

**Use case**: Traditional signal processing, specialized audio networks

### 2. Mel-Spectrogram (Default)

**Format**: 2D frequency-time representation (like an image)
**Shape**: `(n_mels, time_frames, channels)`
**Example**: `(64, 50, 2)` = 64 mel bins, 50 time frames (~10ms each), stereo

```
Mel-Spectrogram is a "photograph" of sound:

High Freq ┌─────────────────┐
          │█░░░░█░░░░░░░░░░│  Each pixel = power at
          │██░░██░░░░░░░░░░│  (frequency, time) point
          │███░███░░░░█░░░░│
Low  Freq │████████░░░█░░░░│  Brighter = louder
          └─────────────────┘
            Time →
```

**Advantages**:
- ✓ Compact: 6,400 features vs 22,050 (6.9x smaller)
- ✓ Works well with CNNs (treat as image)
- ✓ Mimics human hearing (mel scale)
- ✓ Captures important spectral features

**Disadvantages**:
- ✗ Some frequency resolution lost
- ✗ Phase information lost (but often not needed)

**Use case**: Deep learning (THIS IS WHAT PPO USES!)

### Neural Network Input

When the agent receives audio, here's what happens:

```
┌──────────────────────┐
│ Mel-Spectrogram      │
│ (64, 50, 2)          │  ← Agent's audio observation
│                      │
│ [64 mel bins] ×      │
│ [50 time frames] ×   │
│ [2 channels]         │
└──────────┬───────────┘
           │
           │ Flatten to 1D vector
           ▼
    ┌────────────┐
    │   6,400    │  ← Fed to neural network
    └────────────┘
```

The agent sees audio as a 2D image where:
- **X-axis** = time (50 frames, ~10ms each)
- **Y-axis** = frequency (64 mel bins, logarithmic scale)
- **Color** = intensity/power (in dB)
- **Channels** = left and right ear (binaural)

### Binaural Audio (Spatial)

The agent receives **two channels** (left and right ear), which encode:

1. **Interaural Time Difference (ITD)**
   - Sound reaches one ear slightly before the other
   - Helps localize sound source azimuth (left/right)

2. **Interaural Level Difference (ILD)**
   - Sound is louder in one ear due to head shadowing
   - Also helps with azimuth, especially at high frequencies

3. **Spectral Cues**
   - Different frequencies arrive at different times
   - Helps with elevation (up/down)

```
     Robot with Binaural Ears
          ╔═══╗
    Left →║ R ║← Right
    Ear   ║   ║  Ear
          ╚═══╝

Sound Source
    (╯°□°)╯ ♪♫
      │
      └──→ Different arrival times + levels
           encode 3D position!
```

### Visualization Tool

To see how agents perceive audio, run:

```bash
PYTHONPATH=. python3 scripts/visualization/visualize_audio_observation.py
```

This generates:
- `outputs/audio_perception_visualization.png` - 4-panel comparison of raw vs mel-spectrogram
- `outputs/ppo_architecture_diagram.png` - Network architecture showing audio processing

---

## Video Rendering During Training

### Overview

Video rendering lets you visualize what the agent is doing during training. This is crucial for:
- Debugging agent behavior
- Understanding learning progress
- Creating demos and publications
- Identifying failure modes

### How It Works

The `VideoRecorderCallback` captures frames from MuJoCo during evaluation episodes and saves them as MP4 videos.

```
Training Loop
    │
    ├──→ Every 50,000 steps
    │    │
    │    ├──→ Run evaluation episode
    │    │    │
    │    │    ├──→ Capture MuJoCo frames
    │    │    ├──→ Add info overlay
    │    │    └──→ Save as MP4
    │    │
    │    └──→ Continue training
    │
    └──→ Repeat
```

### Video Features

Each video includes:
- **3D rendering** of the data center environment
- **Robot movement** from evaluation episode
- **Info overlay**:
  - Current step count
  - Cumulative reward
  - Distance to goal
- **Camera positioning**: 45° angle showing full scene

### Using Video Recording

**Enable video recording** during training:

```bash
# Record videos every 50,000 steps
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 1000000 \
  --n-envs 4 \
  --record-video \
  --video-freq 50000
```

**Arguments**:
- `--record-video` - Enable video recording
- `--video-freq N` - Record video every N timesteps (default: 50000)

**Output**: Videos saved to `outputs/ppo_datacenter/videos/`
- `training_00050000_steps.mp4` - At 50k steps
- `training_00100000_steps.mp4` - At 100k steps
- `training_00150000_steps.mp4` - At 150k steps
- etc.

### Video Specifications

- **Resolution**: 480×480 pixels
- **FPS**: 10 (matches control frequency)
- **Codec**: MP4 (H.264)
- **Duration**: Up to 500 frames (50 seconds max)
- **Camera**: Free camera with azimuth=135°, elevation=-30°, distance=35m

### Performance Impact

**Important**: Video rendering slows down training!

- **Without video**: ~8 FPS on CPU
- **With video**: ~3-4 FPS during recording episodes
- **Recommendation**: Use larger `--video-freq` (e.g., 100000) to minimize impact

**Training time estimate**:
- 1M timesteps without video: ~35 hours on CPU
- 1M timesteps with video every 50k steps: ~38 hours on CPU

### Manual Video Recording

To record a single video from a trained agent:

```python
from stable_baselines3 import PPO
from physonic.envs.data_center_nav import DataCenterNavigationEnv
import mujoco
import cv2

# Load trained model
model = PPO.load("outputs/ppo_datacenter/models/best_model.zip")

# Create environment
env = DataCenterNavigationEnv(seed=42)

# Setup renderer
renderer = mujoco.Renderer(env.model, width=480, height=480)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.azimuth = 135
camera.elevation = -30
camera.distance = 35
camera.lookat[:] = [15, 12.5, 1.0]

# Record episode
obs, _ = env.reset()
out = cv2.VideoWriter('episode.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (480, 480))

for step in range(500):
    # Render frame
    mujoco.mjv_updateScene(env.model, env.data, mujoco.MjvOption(), None, camera,
                          mujoco.mjtCatBit.mjCAT_ALL, renderer.scene)
    pixels = renderer.render()
    out.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))

    # Take action
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

out.release()
renderer.close()
```

---

## Examples

### Example 1: Visualize Audio Perception

```bash
# Generate audio perception visualization
PYTHONPATH=. python3 scripts/visualization/visualize_audio_observation.py

# View results
open outputs/audio_perception_visualization.png
open outputs/ppo_architecture_diagram.png
```

**Output**:
```
======================================================================
Audio Perception Visualization
======================================================================

Observation shapes:
  Mel-spectrogram: (64, 50, 2) (n_mels, time_frames, channels)
  Raw audio: (22050, 2) (samples, channels)
  Sample rate: 44100 Hz
  Duration: 0.5 seconds

✓ Saved visualization to: outputs/audio_perception_visualization.png
✓ Saved architecture diagram to: outputs/ppo_architecture_diagram.png

Summary:
✓ Raw audio: 22,050 samples × 2 channels = 44,100 numbers
✓ Mel-spectrogram: 64 bins × 50 frames × 2 channels = 6,400 numbers
✓ Compression ratio: 6.9x smaller

The agent sees audio as a 2D image (like a photograph of sound)
where X-axis = time, Y-axis = frequency, and color = intensity.
```

### Example 2: Train with Video Recording

```bash
# Short training run with video (testing)
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 10000 \
  --n-envs 1 \
  --record-video \
  --video-freq 5000
```

**Output**:
```
Setting up callbacks...
Video recording enabled (every 5,000 steps)
Starting training...

[At 5,000 steps]
Recording video: outputs/ppo_datacenter/videos/training_00005000_steps.mp4
  Recorded 500 frames
  Total reward: -62.3
  Result: FAILED
  Saved to: outputs/ppo_datacenter/videos/training_00005000_steps.mp4

[At 10,000 steps]
Recording video: outputs/ppo_datacenter/videos/training_00010000_steps.mp4
  Recorded 245 frames
  Total reward: 3.2
  Result: SUCCESS
  Saved to: outputs/ppo_datacenter/videos/training_00010000_steps.mp4
```

### Example 3: Full Training with Videos

```bash
# Full 1M timestep training with videos every 100k steps
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 1000000 \
  --n-envs 4 \
  --eval-freq 10000 \
  --save-freq 50000 \
  --record-video \
  --video-freq 100000 \
  --device cpu
```

This will create 10 videos showing training progress at:
- 100k steps (early, random behavior)
- 200k steps (starting to learn)
- 300k steps (improving navigation)
- ...
- 1M steps (final trained behavior)

---

## Troubleshooting

### Video Recording Issues

**Issue**: `ModuleNotFoundError: No module named 'cv2'`
**Solution**: Install OpenCV
```bash
pip install opencv-python
```

**Issue**: Videos are corrupted or won't play
**Solution**: Check codec support, try different player (VLC)

**Issue**: Recording is too slow
**Solution**: Increase `--video-freq` or disable video recording

### Audio Observation Issues

**Issue**: Mel-spectrogram is all zeros
**Solution**: This is expected in the current placeholder implementation. Audio rendering will be added in future updates.

**Issue**: `UserWarning: Box low's precision lowered`
**Solution**: This is harmless, Gymnasium converts float64 to float32 automatically.

---

## Implementation Notes

### Current Limitations

1. **Audio is placeholder**: Currently returns zeros
   - TODO: Implement actual audio rendering with moving sound tracker
   - TODO: Add time-varying HRTF for moving sources

2. **Video quality**: 480p resolution
   - Can be increased but slows training significantly
   - Higher resolutions recommended for publication videos only

3. **No camera control during recording**
   - Fixed camera angle (135° azimuth, -30° elevation)
   - Future: Add multiple camera angles or tracking camera

### Future Enhancements

- [ ] Real-time audio rendering with continuous sounds
- [ ] Multiple camera angles (front, side, top, tracking)
- [ ] Overlay showing mel-spectrogram alongside video
- [ ] Audio waveform overlay
- [ ] Goal marker visualization in video
- [ ] Trajectory trail showing robot path
- [ ] Side-by-side comparison videos (random vs trained agent)

---

## References

- **Mel-spectrogram**: Used in speech recognition (DeepSpeech, Wav2Vec) and audio classification
- **Binaural audio**: HRTF-based spatial audio rendering (MIT KEMAR dataset)
- **Video rendering**: MuJoCo offscreen rendering API
- **PPO architecture**: Stable-Baselines3 MultiInputPolicy with Dict observations

---

**Last Updated**: October 2026
**Physonic Version**: 1.0 (Phase 3C)
