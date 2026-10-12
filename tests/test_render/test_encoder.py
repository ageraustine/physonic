"""Tests for video encoding."""

import pytest
import numpy as np
from pathlib import Path
import cv2

from physonic.render.encoder import VideoEncoder, encode_video


class TestVideoEncoder:
    """Test VideoEncoder class."""

    def test_create_encoder(self, tmp_path):
        """Test creating video encoder."""
        output_path = tmp_path / "test.mp4"
        encoder = VideoEncoder(output_path, fps=30, width=640, height=480)

        assert encoder.output_path == output_path
        assert encoder.fps == 30
        assert encoder.width == 640
        assert encoder.height == 480
        assert encoder.frame_count == 0
        assert encoder.is_open

        encoder.close()
        assert not encoder.is_open

    def test_context_manager(self, tmp_path):
        """Test using encoder as context manager."""
        output_path = tmp_path / "test.mp4"

        with VideoEncoder(output_path, fps=30, width=640, height=480) as encoder:
            assert encoder.is_open

        assert not encoder.is_open

    def test_write_frame(self, tmp_path):
        """Test writing a single frame."""
        output_path = tmp_path / "test.mp4"

        # Create a simple RGB frame (red image)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Red channel

        with VideoEncoder(output_path, fps=30, width=640, height=480) as encoder:
            encoder.write(frame)
            assert encoder.frame_count == 1

        # Verify file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_write_multiple_frames(self, tmp_path):
        """Test writing multiple frames."""
        output_path = tmp_path / "test.mp4"

        with VideoEncoder(output_path, fps=30, width=640, height=480) as encoder:
            for i in range(10):
                # Create frame with intensity varying
                frame = np.ones((480, 640, 3), dtype=np.uint8) * (i * 25)
                encoder.write(frame)

            assert encoder.frame_count == 10

        assert output_path.exists()

    def test_wrong_frame_shape(self, tmp_path):
        """Test that wrong frame shape raises error."""
        output_path = tmp_path / "test.mp4"

        with VideoEncoder(output_path, fps=30, width=640, height=480) as encoder:
            # Wrong dimensions
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            with pytest.raises(ValueError, match="Frame shape"):
                encoder.write(frame)

    def test_wrong_frame_dtype(self, tmp_path):
        """Test that wrong frame dtype raises error."""
        output_path = tmp_path / "test.mp4"

        with VideoEncoder(output_path, fps=30, width=640, height=480) as encoder:
            # Wrong dtype
            frame = np.zeros((480, 640, 3), dtype=np.float32)
            with pytest.raises(ValueError, match="must be uint8"):
                encoder.write(frame)

    def test_write_after_close(self, tmp_path):
        """Test that writing after close raises error."""
        output_path = tmp_path / "test.mp4"
        encoder = VideoEncoder(output_path, fps=30, width=640, height=480)
        encoder.close()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="closed"):
            encoder.write(frame)

    def test_create_output_directory(self, tmp_path):
        """Test that encoder creates output directory if needed."""
        output_path = tmp_path / "subdir" / "nested" / "test.mp4"

        # Directory doesn't exist yet
        assert not output_path.parent.exists()

        encoder = VideoEncoder(output_path, fps=30, width=640, height=480)
        encoder.close()

        # Directory should be created
        assert output_path.parent.exists()

    def test_repr(self, tmp_path):
        """Test string representation."""
        output_path = tmp_path / "test.mp4"
        encoder = VideoEncoder(output_path, fps=30, width=640, height=480)

        repr_str = repr(encoder)
        assert "VideoEncoder" in repr_str
        assert "fps=30" in repr_str
        assert "640x480" in repr_str
        assert "frames=0" in repr_str

        encoder.close()

    def test_alternative_codec(self, tmp_path):
        """Test using alternative codec."""
        output_path = tmp_path / "test.mp4"

        # Try xvid codec (more commonly available)
        try:
            encoder = VideoEncoder(
                output_path, fps=30, width=640, height=480, codec="xvid"
            )
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            encoder.write(frame)
            encoder.close()
            assert output_path.exists()
        except RuntimeError:
            # Codec not available - skip test
            pytest.skip("xvid codec not available")


class TestEncodeVideo:
    """Test encode_video convenience function."""

    def test_encode_frames(self, tmp_path):
        """Test encoding a list of frames."""
        output_path = tmp_path / "test.mp4"

        # Create 30 frames (1 second at 30 fps)
        frames = []
        for i in range(30):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * (i * 8)
            frames.append(frame)

        result_path = encode_video(frames, output_path, fps=30)

        assert result_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_encode_empty_list(self, tmp_path):
        """Test that encoding empty list raises error."""
        output_path = tmp_path / "test.mp4"

        with pytest.raises(ValueError, match="empty"):
            encode_video([], output_path)

    def test_encode_inconsistent_sizes(self, tmp_path):
        """Test that inconsistent frame sizes raise error."""
        output_path = tmp_path / "test.mp4"

        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((240, 320, 3), dtype=np.uint8),  # Different size
        ]

        with pytest.raises(ValueError, match="inconsistent size"):
            encode_video(frames, output_path)

    def test_encode_invalid_shape(self, tmp_path):
        """Test that invalid frame shape raises error."""
        output_path = tmp_path / "test.mp4"

        # Grayscale frames (only 2D)
        frames = [np.zeros((480, 640), dtype=np.uint8)]

        with pytest.raises(ValueError, match="must be .* arrays"):
            encode_video(frames, output_path)

    def test_encode_small_resolution(self, tmp_path):
        """Test encoding small resolution video."""
        output_path = tmp_path / "small.mp4"

        # Create 10 small frames
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(10)]

        result_path = encode_video(frames, output_path, fps=10)

        assert result_path.exists()

    def test_encode_high_fps(self, tmp_path):
        """Test encoding high frame rate video."""
        output_path = tmp_path / "highfps.mp4"

        # Create 60 frames at 60 fps (1 second)
        frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(60)]

        result_path = encode_video(frames, output_path, fps=60)

        assert result_path.exists()


@pytest.mark.integration
class TestVideoIntegration:
    """Integration tests for video encoding."""

    def test_read_encoded_video(self, tmp_path):
        """Test that encoded video can be read back."""
        output_path = tmp_path / "test.mp4"

        # Create frames with gradient pattern
        frames = []
        for i in range(30):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 8  # Red channel increases
            frames.append(frame)

        encode_video(frames, output_path, fps=30)

        # Read back with OpenCV
        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened()

        # Check properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        assert fps == 30
        assert width == 100
        assert height == 100
        assert frame_count == 30

        cap.release()

    def test_rgb_to_bgr_conversion(self, tmp_path):
        """Test that RGB frames are correctly converted to BGR."""
        output_path = tmp_path / "test.mp4"

        # Create a pure red frame in RGB
        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_rgb[:, :, 0] = 255  # Red channel in RGB

        encode_video([frame_rgb], output_path, fps=1)

        # Read back (OpenCV returns BGR)
        cap = cv2.VideoCapture(str(output_path))
        ret, frame_bgr = cap.read()
        cap.release()

        assert ret
        # In BGR, red should be in channel 2
        # Allow some lossy compression tolerance
        assert np.mean(frame_bgr[:, :, 2]) > 200  # Red channel (BGR)
        assert np.mean(frame_bgr[:, :, 0]) < 50   # Blue channel
