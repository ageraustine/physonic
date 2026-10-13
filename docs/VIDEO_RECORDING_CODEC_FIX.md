# Video Recording Codec Fix

## Problem

Video files were not being created during PPO training, even though the callbacks reported "Saved training rollout" messages.

## Root Cause

**OpenCV Backend Incompatibility on macOS**

The `opencv-python` package on macOS is built with **AVFoundation** (Apple's native video framework) instead of FFmpeg:

```
Video I/O:
  FFMPEG:        NO
  AVFoundation:  YES
```

AVFoundation has different codec compatibility than FFmpeg. The 'mp4v' codec that was originally used in the callbacks doesn't work reliably with AVFoundation's VideoWriter, causing it to fail silently.

## Solution

**Use 'avc1' (H.264) codec instead of 'mp4v'**

The H.264 codec ('avc1') is natively supported by AVFoundation and works reliably on macOS.

### Files Modified

#### 1. `scripts/training/video_recorder_callback.py` (line 111)

**Before:**
```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
```

**After:**
```python
# Use 'avc1' (H.264) for macOS AVFoundation compatibility
fourcc = cv2.VideoWriter_fourcc(*'avc1')
```

#### 2. `scripts/training/training_monitor_callback.py` (line 187)

**Before:**
```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
```

**After:**
```python
# Use 'avc1' (H.264) for macOS AVFoundation compatibility
fourcc = cv2.VideoWriter_fourcc(*'avc1')
```

#### 3. `scripts/training/test_video_recording.py`

Updated to prioritize 'avc1' codec and test both .mp4 and .mov extensions for maximum compatibility.

## Verification

### Test Results

After the fix, video recording works correctly:

```bash
$ python3 scripts/training/test_video_recording.py

✓ SUCCESS: Video recording works!
✓ Test video saved to: outputs/video_test/test_video.mp4
  Recommended codec for training: avc1
```

### Video Properties

```bash
$ ffprobe outputs/video_test_ppo/eval_videos/training_00001500_steps.mp4

codec_name=h264
width=480
height=480
r_frame_rate=10/1
duration=50.000000
nb_frames=500
```

Both evaluation and training videos are created successfully:
- **Evaluation video**: 500 frames (50 seconds) at 992KB
- **Training video**: 100 frames (10 seconds) at 210KB

## OpenCV Backend Detection

To check which backend your OpenCV is using:

```bash
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -A 10 "Video I/O"
```

### Codec Compatibility by Backend

| Backend      | mp4v | avc1 | MJPG | XVID |
|--------------|------|------|------|------|
| FFmpeg       | ✓    | ✓    | ✓    | ✓    |
| AVFoundation | ✗    | ✓    | ✓    | ?    |
| GStreamer    | ✓    | ✓    | ✓    | ✓    |

**Recommendation**: Always use 'avc1' (H.264) for maximum compatibility across all backends.

## Known Warnings (Can Be Ignored)

When using AVFoundation, you may see these warnings:

```
WARNING: -finishWriting should not be called on the main thread.
OpenCV: AVF: waiting to write video data.
```

These are AVFoundation threading warnings and don't indicate a problem. The videos are still created successfully.

## Alternative Solutions (If Issues Persist)

If 'avc1' doesn't work on your system:

1. **Try 'MJPG' codec** (Motion JPEG):
   ```python
   fourcc = cv2.VideoWriter_fourcc(*'MJPG')
   ```
   - Larger file sizes but more universal compatibility

2. **Use .mov extension** instead of .mp4:
   ```python
   video_path = Path("video.mov")  # instead of .mp4
   ```

3. **Install opencv-contrib-python** (includes more codecs):
   ```bash
   pip uninstall opencv-python
   pip install opencv-contrib-python
   ```

4. **Install FFmpeg-based OpenCV** (Linux/Windows):
   ```bash
   # Homebrew (macOS)
   brew install opencv

   # Or build from source with FFmpeg support
   ```

## Summary

- **Problem**: Videos not created due to codec incompatibility
- **Cause**: OpenCV on macOS uses AVFoundation, which doesn't support 'mp4v'
- **Fix**: Changed codec from 'mp4v' to 'avc1' (H.264)
- **Result**: ✓ Videos now created successfully (verified with ffprobe)
- **Status**: Issue resolved

## Training with Video Recording

Now you can use video recording during training:

```bash
# Evaluation videos (always work with any n-envs)
python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 100000 \
  --n-envs 4 \
  --record-video \
  --video-freq 10000

# Training videos (require n-envs=1)
python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 100000 \
  --n-envs 1 \
  --monitor-training \
  --monitor-freq 5000

# Both types of videos
python3 scripts/training/train_ppo_data_center.py \
  --total-timesteps 100000 \
  --n-envs 1 \
  --record-video \
  --monitor-training \
  --video-freq 10000 \
  --monitor-freq 5000
```

Videos are saved to:
- **Evaluation**: `outputs/ppo_datacenter/eval_videos/`
- **Training**: `outputs/ppo_datacenter/training_videos/`
