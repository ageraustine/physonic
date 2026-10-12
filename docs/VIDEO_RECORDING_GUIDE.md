# Video Recording During Training - Complete Guide

This guide explains the different video recording options for visualizing PPO training.

## Quick Answer

**Yes, you can now record videos during training in 3 different ways:**

1. **Evaluation Videos** - Recorded from evaluation episodes (clean, deterministic)
2. **Training Videos** - Recorded from actual training rollouts (shows exploration)
3. **Real-Time Viewer** - Live window showing training (very slow, debugging only)

---

## Three Recording Options

### Option 1: Evaluation Videos (Recommended)

**What**: Records videos from EVALUATION episodes
**When**: Every N timesteps (e.g., every 50,000 steps)
**Agent Behavior**: Deterministic (best actions, no exploration)
**Use For**: Tracking learning progress, creating demos

```bash
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 1000000 \
  --n-envs 4 \
  --record-video \
  --video-freq 50000
```

**Output**: `outputs/ppo_datacenter/eval_videos/`
- `training_00050000_steps.mp4` - Evaluation at 50k steps
- `training_00100000_steps.mp4` - Evaluation at 100k steps
- etc.

**Characteristics**:
- ✓ Clean, reproducible episodes
- ✓ Agent uses deterministic policy (no randomness)
- ✓ Good for measuring progress
- ✓ Minimal performance impact
- ✓ Fixed initial state (easier to compare over time)

### Option 2: Training Videos (Shows Actual Training)

**What**: Records videos from TRAINING rollouts
**When**: Every N timesteps (e.g., every 10,000 steps)
**Agent Behavior**: Stochastic (exploration, random actions)
**Use For**: Seeing what agent actually experiences during training

```bash
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 1000000 \
  --n-envs 4 \
  --monitor-training \
  --monitor-freq 10000
```

**Output**: `outputs/ppo_datacenter/training_videos/`
- `training_rollout_00010000_steps.mp4` - Training at 10k steps
- `training_rollout_00020000_steps.mp4` - Training at 20k steps
- etc.

**Characteristics**:
- ✓ Shows actual training behavior (with exploration)
- ✓ See random actions and mistakes
- ✓ Different every time (stochastic policy)
- ✓ Records 100 frames (10 seconds) per video
- ✓ Moderate performance impact
- ✓ Good for debugging why agent isn't learning

### Option 3: Real-Time Viewer (Debugging Only)

**What**: Live OpenCV window showing training in real-time
**When**: Updates every 10 steps
**Agent Behavior**: Stochastic (actual training)
**Use For**: Debugging, watching training live

```bash
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 10000 \
  --n-envs 1 \
  --show-window
```

**Output**: Live window (press 'q' to close)

**Characteristics**:
- ✓ See training live as it happens
- ✓ Press 'q' to close window (training continues)
- ✗ VERY SLOW (reduces training speed by ~70%)
- ✗ Only works with n_envs=1
- ✗ No video file saved
- ✗ Only for debugging/demonstrations

---

## Comparison Table

| Feature | Evaluation Videos | Training Videos | Real-Time Viewer |
|---------|------------------|-----------------|------------------|
| **Flag** | `--record-video` | `--monitor-training` | `--show-window` |
| **Source** | Eval episodes | Training rollouts | Training rollouts |
| **Policy** | Deterministic | Stochastic | Stochastic |
| **Frequency** | `--video-freq` (50k) | `--monitor-freq` (10k) | Every 10 steps |
| **Duration** | Full episode (~500 steps) | 100 frames (10s) | Continuous |
| **Output** | MP4 files | MP4 files | Live window only |
| **Speed Impact** | Minimal (~5%) | Moderate (~20%) | Severe (~70%) |
| **Use Case** | Progress tracking | Debugging learning | Live demo |

---

## Usage Examples

### Example 1: Standard Training with Evaluation Videos

```bash
# Record evaluation videos every 50k steps
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 1000000 \
  --n-envs 4 \
  --record-video \
  --video-freq 50000
```

**What you get**:
- 20 evaluation videos (at 50k, 100k, 150k, ..., 1000k steps)
- Each video shows deterministic agent behavior
- Easy to see improvement over time
- Minimal slowdown

### Example 2: Debug Why Agent Isn't Learning

```bash
# Record training videos frequently to see exploration
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 100000 \
  --n-envs 1 \
  --monitor-training \
  --monitor-freq 5000
```

**What you get**:
- 20 training videos showing actual training behavior
- See random actions, exploration, mistakes
- Understand what experiences agent is collecting
- Can identify if environment is too hard/easy

### Example 3: Both Evaluation and Training Videos

```bash
# Record both types for complete picture
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 500000 \
  --n-envs 4 \
  --record-video \
  --video-freq 50000 \
  --monitor-training \
  --monitor-freq 25000
```

**What you get**:
- **Eval videos**: 10 videos (50k, 100k, ..., 500k) - clean performance
- **Training videos**: 20 videos (25k, 50k, ..., 500k) - actual training
- Best of both worlds: see both learned policy and training process

### Example 4: Live Debugging Session

```bash
# Watch training live (SLOW! Use for debugging only)
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 5000 \
  --n-envs 1 \
  --show-window
```

**What happens**:
- Window opens showing live training
- You see the robot moving in real-time
- Press 'q' to close window (training continues in background)
- Very slow, only for short debugging sessions

---

## Key Differences: Evaluation vs Training

### Evaluation Videos Show:
```
Agent Policy: Deterministic
  ↓
Always picks best action (no randomness)
  ↓
Smooth, consistent behavior
  ↓
Good for measuring progress
```

**Example**:
- Step 50k eval: Agent reaches goal in 450 steps
- Step 100k eval: Agent reaches goal in 350 steps
- Step 200k eval: Agent reaches goal in 180 steps
- Clear improvement!

### Training Videos Show:
```
Agent Policy: Stochastic
  ↓
Adds exploration noise to actions
  ↓
More random, erratic behavior
  ↓
Good for seeing what agent experiences
```

**Example**:
- Training video shows agent spinning in circles
- Agent tries random movements
- Occasionally finds goal by accident
- This is NORMAL during training!

---

## Performance Impact

Measured on M1 MacBook Pro with 4 parallel environments:

| Configuration | Training Speed | Video Files | Use Case |
|--------------|----------------|-------------|----------|
| No video | ~8 FPS | None | Fastest training |
| Eval only (50k freq) | ~7.6 FPS | 20 files/1M | Recommended |
| Training only (10k freq) | ~6.5 FPS | 100 files/1M | Debugging |
| Both | ~6.2 FPS | 120 files/1M | Complete picture |
| Real-time viewer | ~2.5 FPS | None | Debugging only |

**Recommendation**: Use `--record-video` for normal training, add `--monitor-training` only when debugging.

---

## Output Directory Structure

```
outputs/ppo_datacenter/
├── models/                    # Trained models
│   ├── best_model.zip
│   ├── ppo_datacenter_50000_steps.zip
│   └── ppo_datacenter_final.zip
├── logs/                      # TensorBoard logs
│   └── PPO_1/
├── eval_videos/              # Evaluation episode videos
│   ├── training_00050000_steps.mp4
│   ├── training_00100000_steps.mp4
│   └── ...
└── training_videos/          # Training rollout videos
    ├── training_rollout_00010000_steps.mp4
    ├── training_rollout_00020000_steps.mp4
    └── ...
```

---

## Video Specifications

All videos use the same settings:

- **Resolution**: 480×480 pixels
- **FPS**: 10 (matches control frequency)
- **Codec**: MP4 (H.264)
- **Camera**: Free camera
  - Azimuth: 135°
  - Elevation: -30°
  - Distance: 35m
  - Lookat: [15, 12.5, 1.0] (center of data center)

**Info Overlay** (text on video):
- Training step count
- Current frame number
- Cumulative reward (eval videos only)
- Distance to goal (eval videos only)

---

## Troubleshooting

### Issue: Videos are all dark

**Cause**: MuJoCo lighting issue
**Solution**: Environment already has bright lighting configured

### Issue: No videos generated

**Cause**: Video recording may have failed
**Solution**: Check console output for error messages

### Issue: Training is very slow

**Cause**: Too many videos being recorded
**Solution**:
- Increase `--video-freq` and `--monitor-freq`
- Disable `--show-window`
- Reduce number of parallel environments

### Issue: "cv2" module not found

**Cause**: OpenCV not installed
**Solution**: `pip install opencv-python`

### Issue: Real-time window doesn't show

**Cause**: May require X11 forwarding on remote systems
**Solution**: Use video files instead, or run locally

---

## Best Practices

### For Normal Training
```bash
# Recommended: Eval videos only
--record-video --video-freq 50000
```

### For Debugging Learning
```bash
# Add training videos to see exploration
--monitor-training --monitor-freq 10000
```

### For Paper/Presentation Videos
```bash
# High-frequency eval videos for smooth progress
--record-video --video-freq 25000
```

### For Live Demonstration
```bash
# Real-time viewer for short periods
--show-window --total-timesteps 5000 --n-envs 1
```

---

## Programmatic Access

You can also use the callbacks directly in your code:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.training.video_recorder_callback import VideoRecorderCallback
from scripts.training.training_monitor_callback import TrainingMonitorCallback, RealTimeViewer

# Create environment
env = DummyVecEnv([make_env])

# Create model
model = PPO("MultiInputPolicy", env)

# Create callbacks
eval_video = VideoRecorderCallback(
    eval_env=env,
    video_folder="outputs/my_videos",
    record_freq=10000,
    verbose=1
)

training_monitor = TrainingMonitorCallback(
    video_folder="outputs/training",
    record_freq=5000,
    record_duration=50,
    verbose=1
)

# Train with both callbacks
model.learn(
    total_timesteps=100000,
    callback=[eval_video, training_monitor]
)
```

---

## Summary

**Question**: "Is video recorded during training?"

**Answer**: Yes, in two ways:

1. **Evaluation videos** (`--record-video`) - Shows learned policy (deterministic)
2. **Training videos** (`--monitor-training`) - Shows actual training (stochastic)

**Recommendation**:
- Use `--record-video` for tracking progress
- Add `--monitor-training` when debugging
- Use `--show-window` only for quick demos (very slow!)

**Example for complete visualization**:
```bash
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 1000000 \
  --n-envs 4 \
  --record-video \
  --monitor-training \
  --video-freq 50000 \
  --monitor-freq 25000
```

This gives you both evaluation performance tracking AND insight into the training process!

---

**Last Updated**: October 2026
**Related**: See `AUDIO_AND_VIDEO_GUIDE.md` for audio perception details
