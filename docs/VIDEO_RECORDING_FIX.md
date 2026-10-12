# Video Recording Fix - SubprocVecEnv Issue

## Problem

You saw this warning:
```
Warning: Could not initialize renderer: 'SubprocVecEnv' object has no attribute 'envs'
```

## Root Cause

**SubprocVecEnv** (used for parallel training with `--n-envs > 1`) runs environments in **separate processes**, so you can't directly access the MuJoCo model for rendering.

**DummyVecEnv** (used for single environment with `--n-envs 1`) runs everything in the **main process**, so rendering works.

## Solution

### ✅ Fixed: Evaluation Videos ALWAYS Work

Evaluation videos now work regardless of `--n-envs` setting because the evaluation environment is always `DummyVecEnv`.

```bash
# This NOW WORKS even with parallel training
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 1000000 \
  --n-envs 4 \
  --record-video \
  --video-freq 50000
```

**Output**: Evaluation videos will be recorded successfully! ✓

### ⚠️ Limitation: Training Videos Require n-envs=1

Training videos (`--monitor-training`) and real-time viewer (`--show-window`) only work with single environment.

```bash
# Training videos - REQUIRES n-envs=1
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 100000 \
  --n-envs 1 \
  --monitor-training \
  --monitor-freq 10000
```

If you try with `--n-envs 4`, you'll get a clear warning:
```
⚠️  WARNING: Training monitor requires --n-envs 1
   Training videos will NOT be recorded with parallel environments
   Use --record-video for evaluation videos instead
```

## Recommended Configurations

### For Fast Training (Parallel Environments)

```bash
# Use 4 parallel environments for speed
# Record EVALUATION videos only
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 1000000 \
  --n-envs 4 \
  --record-video \
  --video-freq 50000
```

**Result**:
- ✓ Fast training (4x speedup)
- ✓ Evaluation videos recorded
- ✗ No training videos (requires single environment)

### For Complete Visualization (Single Environment)

```bash
# Use 1 environment to enable all video recording
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 500000 \
  --n-envs 1 \
  --record-video \
  --monitor-training \
  --video-freq 50000 \
  --monitor-freq 25000
```

**Result**:
- ✓ Evaluation videos
- ✓ Training videos
- ✗ Slower training (no parallelization)

### For Debugging (Live Viewer)

```bash
# Use 1 environment with live window
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 10000 \
  --n-envs 1 \
  --show-window
```

**Result**:
- ✓ Live window showing training
- ✓ Press 'q' to close window
- ✗ Very slow (70% slower)

## Why This Happens

### DummyVecEnv (n-envs=1)
```
Main Process
    │
    ├── Environment
    ├── MuJoCo Model ← Can access directly
    └── Renderer ← Can create
```

### SubprocVecEnv (n-envs>1)
```
Main Process
    │
    ├── Subprocess 1 → Environment 1 → MuJoCo Model ← Can't access from main!
    ├── Subprocess 2 → Environment 2 → MuJoCo Model ← Can't access from main!
    ├── Subprocess 3 → Environment 3 → MuJoCo Model ← Can't access from main!
    └── Subprocess 4 → Environment 4 → MuJoCo Model ← Can't access from main!
```

With `SubprocVecEnv`, each environment runs in its own process for parallelization, but you can't access the MuJoCo models from the main process for rendering.

## Summary Table

| Feature | n-envs=1 | n-envs>1 |
|---------|----------|----------|
| **Training Speed** | Slow (baseline) | Fast (4x with n-envs=4) |
| **Evaluation Videos** | ✓ Works | ✓ Works (FIXED!) |
| **Training Videos** | ✓ Works | ✗ Not available |
| **Real-Time Viewer** | ✓ Works | ✗ Not available |
| **Best For** | Debugging, visualization | Fast training |

## Recommendations

1. **Normal training**: Use `--n-envs 4` with `--record-video` for fast training + eval videos
2. **Debugging**: Use `--n-envs 1` with `--monitor-training` to see training behavior
3. **Live demo**: Use `--n-envs 1` with `--show-window` for quick demonstrations

## What Changed in the Fix

1. **Evaluation environment**: Always uses `DummyVecEnv` (even if training uses `SubprocVecEnv`)
2. **Better error messages**: Clear warnings when video recording isn't available
3. **Automatic detection**: Callbacks check environment type and skip gracefully
4. **User warnings**: Clear messages tell you when features won't work

## Example Workflow

### Phase 1: Fast Training (Use Parallel Envs)
```bash
# Train fast with parallel environments
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 1000000 \
  --n-envs 4 \
  --record-video \
  --video-freq 50000 \
  --output-dir outputs/ppo_fast
```

### Phase 2: Debug Learning (Use Single Env)
If agent isn't learning well, debug with training videos:
```bash
# Debug with single environment
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 100000 \
  --n-envs 1 \
  --monitor-training \
  --monitor-freq 10000 \
  --output-dir outputs/ppo_debug
```

### Phase 3: Final Evaluation
Load best model and evaluate:
```bash
PYTHONPATH=. python3 scripts/training/train_ppo_data_center.py \
  --eval-only \
  --load outputs/ppo_fast/models/best_model.zip \
  --n-eval-episodes 20
```

---

**Bottom Line**: Evaluation videos work with any `--n-envs` setting (fixed!), but training videos and real-time viewer require `--n-envs 1` due to Python multiprocessing limitations.
