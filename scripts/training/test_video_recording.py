"""Test video recording to diagnose issues."""

import numpy as np
import cv2
import sys
from pathlib import Path

print("Video Recording Diagnostic Test")
print("=" * 70)

# Test 1: Check OpenCV installation
print("\n1. Checking OpenCV installation...")
try:
    print(f"   ✓ OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"   ✗ OpenCV error: {e}")
    sys.exit(1)

# Test 2: Check available codecs (prioritize AVFoundation-compatible ones)
print("\n2. Checking available codecs...")
print("   OpenCV backend: AVFoundation (macOS native)")
codecs_to_test = [
    ('avc1', 'H.264 (AVC) - AVFoundation preferred'),
    ('mp4v', 'MP4V (MPEG-4)'),
    ('MJPG', 'Motion JPEG'),
    ('XVID', 'XVID'),
]

working_codec = None
for codec_str, codec_name in codecs_to_test:
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec_str)
        print(f"   ✓ {codec_name} ({codec_str}): Available")
        if working_codec is None:
            working_codec = codec_str
    except Exception as e:
        print(f"   ✗ {codec_name} ({codec_str}): Not available - {e}")

if working_codec is None:
    print("\n   ✗ ERROR: No working codecs found!")
    sys.exit(1)

print(f"\n   → Will use: {working_codec}")

# Test 3: Create a simple test video
print("\n3. Creating test video...")
output_dir = Path("outputs/video_test")
output_dir.mkdir(parents=True, exist_ok=True)

# Try both .mp4 and .mov (AVFoundation may prefer .mov)
extensions_to_try = ['.mp4', '.mov']
test_video_path = None
out = None

for ext in extensions_to_try:
    try_path = output_dir / f"test_video{ext}"
    print(f"   Trying {working_codec} with {ext}...")

    fourcc = cv2.VideoWriter_fourcc(*working_codec)
    temp_out = cv2.VideoWriter(str(try_path), fourcc, 10, (480, 480))

    if temp_out.isOpened():
        print(f"   ✓ VideoWriter opened successfully with {ext}")
        test_video_path = try_path
        out = temp_out
        break
    else:
        print(f"   ✗ Failed with {ext}, trying next...")
        temp_out.release()

if out is None or not out.isOpened():
    print(f"\n   ✗ Failed to open VideoWriter with any extension")
    print(f"   This may be an AVFoundation/OpenCV compatibility issue")
    print(f"   Suggestion: Install opencv-python with FFmpeg support:")
    print(f"     pip uninstall opencv-python")
    print(f"     pip install opencv-contrib-python")
    sys.exit(1)

try:
    # Generate 30 test frames (colorful gradient)
    print("   Generating 30 frames...")
    for i in range(30):
        # Create colorful gradient frame
        frame = np.zeros((480, 480, 3), dtype=np.uint8)

        # Horizontal gradient
        for x in range(480):
            frame[:, x] = [
                int((x / 480) * 255),           # Red
                int((i / 30) * 255),             # Green (changes with frame)
                int(((480 - x) / 480) * 255)     # Blue
            ]

        # Add frame number text
        cv2.putText(frame, f"Frame {i+1}/30", (150, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        # Write frame
        out.write(frame)

        if (i + 1) % 10 == 0:
            print(f"   ... wrote {i+1} frames")

    # Release video
    out.release()
    print(f"   ✓ Video released")

    # Check file was created
    if test_video_path.exists():
        file_size = test_video_path.stat().st_size
        print(f"\n   ✓ Video file created: {test_video_path}")
        print(f"   ✓ File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

        if file_size == 0:
            print(f"   ✗ WARNING: File is empty!")
        elif file_size < 1000:
            print(f"   ⚠ WARNING: File is very small, may be corrupted")
        else:
            print(f"   ✓ File size looks good")
    else:
        print(f"\n   ✗ Video file was NOT created")
        sys.exit(1)

except Exception as e:
    print(f"   ✗ Error creating video: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Try to read the video back
print("\n4. Verifying video can be read...")
try:
    cap = cv2.VideoCapture(str(test_video_path))

    if not cap.isOpened():
        print(f"   ✗ Could not open video for reading")
    else:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"   ✓ Video properties:")
        print(f"     - Frames: {frame_count}")
        print(f"     - FPS: {fps}")
        print(f"     - Resolution: {width}×{height}")

        # Read first frame
        ret, frame = cap.read()
        if ret:
            print(f"   ✓ Successfully read first frame")
        else:
            print(f"   ✗ Could not read first frame")

        cap.release()

except Exception as e:
    print(f"   ✗ Error reading video: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

if test_video_path.exists() and test_video_path.stat().st_size > 1000:
    print(f"\n✓ SUCCESS: Video recording works!")
    print(f"✓ Test video saved to: {test_video_path}")
    print(f"\nTry opening it with:")
    print(f"  open {test_video_path}")
    print(f"\nRecommended codec for training: {working_codec}")
else:
    print(f"\n✗ FAILED: Video recording has issues")
    print(f"\nPossible solutions:")
    print(f"  1. Try different codec in VideoWriter")
    print(f"  2. Install ffmpeg: brew install ffmpeg")
    print(f"  3. Reinstall opencv: pip install --force-reinstall opencv-python")
