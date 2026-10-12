"""
Video rendering module for Physonic (Phase 3B).

This module provides MuJoCo-based video rendering capabilities for generating
multi-camera views of physics simulations. Supports both real-time capture and
post-hoc rendering from saved telemetry.

Key components:
- CameraConfig: Camera configuration with position, orientation, FOV
- CameraPreset: Factory methods for common camera setups
- VideoRenderer: MuJoCo rendering engine with multi-camera support
- VideoEncoder: MP4 video encoding using OpenCV
- RenderConfig: Global rendering configuration

Usage:
    from physonic.render import CameraPreset, VideoRenderer, RenderConfig

    # Configure rendering
    config = RenderConfig(
        width=640,
        height=480,
        fps=30,
        cameras=[CameraPreset.fixed_front()]
    )

    # Render from telemetry
    with VideoRenderer(model, config) as renderer:
        frames = renderer.render_from_telemetry(telemetry, duration=2.5)
"""

from physonic.render.camera import CameraConfig, CameraPreset, RenderConfig, get_camera_mode
from physonic.render.renderer import VideoRenderer
from physonic.render.encoder import VideoEncoder, encode_video

__all__ = [
    "CameraConfig",
    "CameraPreset",
    "RenderConfig",
    "VideoRenderer",
    "VideoEncoder",
    "get_camera_mode",
    "encode_video",
]
