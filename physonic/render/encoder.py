"""
Video encoding for MP4 output.

This module provides the VideoEncoder class for encoding RGB frames to
MP4 video files using OpenCV's VideoWriter with H.264 codec.
"""

from pathlib import Path
from typing import List, Union
import numpy as np
import cv2


class VideoEncoder:
    """Encodes RGB frames to MP4 video using OpenCV.

    Uses H.264 codec for efficient compression and broad compatibility.
    Automatically converts RGB frames to BGR for OpenCV.

    Example:
        # Encode a list of frames
        encoder = VideoEncoder("output.mp4", fps=30, width=640, height=480)
        for frame in frames:  # frame is (H, W, 3) RGB uint8
            encoder.write(frame)
        encoder.close()

        # Or use as context manager
        with VideoEncoder("output.mp4", fps=30, width=640, height=480) as enc:
            for frame in frames:
                enc.write(frame)
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        fps: int,
        width: int,
        height: int,
        codec: str = "mp4v"
    ):
        """Initialize video encoder.

        Args:
            output_path: Path to output MP4 file
            fps: Frames per second
            width: Frame width in pixels
            height: Frame height in pixels
            codec: FourCC codec code. Default "mp4v" (MPEG-4).
                   Options: "mp4v", "avc1" (H.264), "h264", "xvid"

        Note:
            H.264 codec may not be available on all systems. If encoding
            fails, try "mp4v" or "xvid" instead.
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.frame_count = 0
        self.writer = None

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Try requested codec, then fallback to alternatives
        codecs_to_try = [codec]
        if codec == "mp4v":
            # Common fallbacks for mp4v
            codecs_to_try.extend(["avc1", "xvid", "mjpg"])
        elif codec not in codecs_to_try:
            codecs_to_try.extend(["mp4v", "xvid"])

        # Remove duplicates while preserving order
        seen = set()
        codecs_to_try = [c for c in codecs_to_try if not (c in seen or seen.add(c))]

        last_error = None
        for trial_codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*trial_codec)
                self.writer = cv2.VideoWriter(
                    str(self.output_path),
                    fourcc,
                    fps,
                    (width, height)  # Note: OpenCV uses (width, height) order
                )

                if self.writer.isOpened():
                    # Success - use this codec
                    self.codec = trial_codec
                    break
                else:
                    last_error = f"Codec '{trial_codec}' failed to open"
            except Exception as e:
                last_error = f"Codec '{trial_codec}' error: {e}"

        if self.writer is None or not self.writer.isOpened():
            raise RuntimeError(
                f"Failed to open video writer for {output_path}. "
                f"Tried codecs: {codecs_to_try}. Last error: {last_error}"
            )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close video file."""
        self.close()

    def write(self, frame: np.ndarray):
        """Write a single frame to video.

        Args:
            frame: RGB frame as (H, W, 3) uint8 array

        Raises:
            ValueError: If frame dimensions don't match configuration
            RuntimeError: If writer is closed or write fails
        """
        if self.writer is None:
            raise RuntimeError("VideoEncoder is closed")

        # Validate frame
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Frame shape {frame.shape} doesn't match "
                f"expected ({self.height}, {self.width}, 3)"
            )

        if frame.dtype != np.uint8:
            raise ValueError(f"Frame must be uint8, got {frame.dtype}")

        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write frame
        self.writer.write(bgr_frame)
        self.frame_count += 1

    def close(self):
        """Close video file and finalize encoding."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    @property
    def is_open(self) -> bool:
        """Check if encoder is open."""
        return self.writer is not None and self.writer.isOpened()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VideoEncoder(path={self.output_path}, "
            f"fps={self.fps}, size={self.width}x{self.height}, "
            f"codec={self.codec}, frames={self.frame_count})"
        )


def encode_video(
    frames: List[np.ndarray],
    output_path: Union[str, Path],
    fps: int = 30,
    codec: str = "mp4v"
) -> Path:
    """Convenience function to encode a list of frames to video.

    Args:
        frames: List of RGB frames as (H, W, 3) uint8 arrays
        output_path: Path to output MP4 file
        fps: Frames per second
        codec: FourCC codec code (default "mp4v")

    Returns:
        Path to created video file

    Raises:
        ValueError: If frames list is empty or frames have inconsistent sizes

    Example:
        frames = [...]  # List of (480, 640, 3) RGB frames
        video_path = encode_video(frames, "output.mp4", fps=30)
        print(f"Saved video to {video_path}")
    """
    if not frames:
        raise ValueError("frames list is empty")

    # Get dimensions from first frame
    first_frame = frames[0]
    if first_frame.ndim != 3 or first_frame.shape[2] != 3:
        raise ValueError(
            f"Frames must be (H, W, 3) arrays, got shape {first_frame.shape}"
        )

    height, width = first_frame.shape[:2]

    # Encode all frames
    with VideoEncoder(output_path, fps, width, height, codec) as encoder:
        for i, frame in enumerate(frames):
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Frame {i} has inconsistent size {frame.shape[:2]}, "
                    f"expected ({height}, {width})"
                )
            encoder.write(frame)

    return Path(output_path)
