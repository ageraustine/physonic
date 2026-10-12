"""
Camera configuration and presets for video rendering.

This module provides camera configuration dataclasses and preset factory methods
for common camera setups (fixed front/side/top, tracking, egocentric).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class CameraConfig:
    """Configuration for a single camera viewpoint.

    Attributes:
        name: Camera identifier (e.g., "front", "side", "tracking_obj")
        type: Camera type - "fixed" (static position) or "tracking" (follows object)
        lookat: 3D point the camera looks at [x, y, z] in meters
        distance: Distance from lookat point to camera in meters
        azimuth: Horizontal rotation in degrees (0=+X axis, 90=+Y axis, counterclockwise)
        elevation: Vertical rotation in degrees (-90=straight down, 0=horizon, 90=straight up)
        fov: Vertical field of view in degrees
        track_object: Optional object name to track (for tracking cameras)

    Example:
        # Fixed camera looking at origin from front
        cam = CameraConfig(
            name="front",
            type="fixed",
            lookat=np.array([0.0, 0.0, 0.25]),
            distance=2.0,
            azimuth=0.0,
            elevation=-15.0,
            fov=45.0
        )
    """
    name: str
    type: str  # "fixed" or "tracking"
    lookat: np.ndarray
    distance: float
    azimuth: float
    elevation: float
    fov: float = 45.0
    track_object: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.type not in ("fixed", "tracking"):
            raise ValueError(f"Camera type must be 'fixed' or 'tracking', got '{self.type}'")
        if self.type == "tracking" and self.track_object is None:
            raise ValueError("Tracking cameras must specify track_object")
        if not isinstance(self.lookat, np.ndarray):
            self.lookat = np.array(self.lookat, dtype=np.float64)
        if self.lookat.shape != (3,):
            raise ValueError(f"lookat must be 3D point, got shape {self.lookat.shape}")
        if self.distance <= 0:
            raise ValueError(f"distance must be positive, got {self.distance}")
        if not (0 < self.fov < 180):
            raise ValueError(f"fov must be in (0, 180) degrees, got {self.fov}")


@dataclass
class RenderConfig:
    """Global rendering configuration.

    Attributes:
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second for video output
        cameras: List of camera configurations to render

    Example:
        config = RenderConfig(
            width=640,
            height=480,
            fps=30,
            cameras=[CameraPreset.fixed_front(), CameraPreset.fixed_side()]
        )
    """
    width: int = 640
    height: int = 480
    fps: int = 30
    cameras: List[CameraConfig] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width and height must be positive, got {self.width}x{self.height}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if not self.cameras:
            raise ValueError("Must specify at least one camera")


class CameraPreset:
    """Factory methods for common camera configurations.

    Provides preset cameras:
    - fixed_front: Front view looking at scene center
    - fixed_side: Side view looking at scene center
    - fixed_top: Top-down view
    - tracking: Camera that follows a specific object
    - egocentric: First-person view attached to object

    All presets use sensible defaults for robotics scenarios:
    - Scene center at ground level (z=0.25m for typical object height)
    - Distances appropriate for 1-2m workspace
    - Slight downward angle for better visibility
    """

    @staticmethod
    def fixed_front(scene_center: Optional[np.ndarray] = None) -> CameraConfig:
        """Fixed front camera looking at scene from -X direction.

        Args:
            scene_center: 3D point to look at. Default: [0, 0, 0.25]

        Returns:
            CameraConfig for fixed front view
        """
        if scene_center is None:
            scene_center = np.array([0.0, 0.0, 0.25])

        return CameraConfig(
            name="front",
            type="fixed",
            lookat=scene_center,
            distance=2.0,
            azimuth=0.0,      # Looking from -X toward +X
            elevation=-15.0,  # Slightly above horizon
            fov=45.0
        )

    @staticmethod
    def fixed_side(scene_center: Optional[np.ndarray] = None) -> CameraConfig:
        """Fixed side camera looking at scene from -Y direction.

        Args:
            scene_center: 3D point to look at. Default: [0, 0, 0.25]

        Returns:
            CameraConfig for fixed side view
        """
        if scene_center is None:
            scene_center = np.array([0.0, 0.0, 0.25])

        return CameraConfig(
            name="side",
            type="fixed",
            lookat=scene_center,
            distance=2.0,
            azimuth=90.0,     # Looking from -Y toward +Y
            elevation=-15.0,  # Slightly above horizon
            fov=45.0
        )

    @staticmethod
    def fixed_top(scene_center: Optional[np.ndarray] = None) -> CameraConfig:
        """Fixed top-down camera looking at scene from above.

        Args:
            scene_center: 3D point to look at. Default: [0, 0, 0.25]

        Returns:
            CameraConfig for fixed top-down view
        """
        if scene_center is None:
            scene_center = np.array([0.0, 0.0, 0.25])

        return CameraConfig(
            name="top",
            type="fixed",
            lookat=scene_center,
            distance=2.5,
            azimuth=0.0,      # Azimuth doesn't matter for top-down
            elevation=89.0,   # Nearly straight down (89° to avoid singularity)
            fov=45.0
        )

    @staticmethod
    def tracking(
        object_name: str,
        initial_lookat: Optional[np.ndarray] = None,
        distance: float = 1.5,
        azimuth: float = 45.0,
        elevation: float = -20.0
    ) -> CameraConfig:
        """Tracking camera that follows a specific object.

        The lookat point will be updated each frame to track the object's position.

        Args:
            object_name: Name of MuJoCo body to track
            initial_lookat: Initial lookat point. Default: [0, 0, 0.25]
            distance: Distance from object in meters
            azimuth: Viewing angle in degrees (0=front, 90=side)
            elevation: Vertical angle in degrees (negative = above horizon)

        Returns:
            CameraConfig for tracking camera
        """
        if initial_lookat is None:
            initial_lookat = np.array([0.0, 0.0, 0.25])

        return CameraConfig(
            name=f"tracking_{object_name}",
            type="tracking",
            lookat=initial_lookat,
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
            fov=45.0,
            track_object=object_name
        )

    @staticmethod
    def egocentric(
        object_name: str,
        initial_lookat: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None
    ) -> CameraConfig:
        """Egocentric (first-person) camera attached to object.

        Camera is positioned at object location with optional offset.
        Useful for robot POV perspectives.

        Args:
            object_name: Name of MuJoCo body to attach to
            initial_lookat: Initial lookat point. Default: [0, 0, 0.25]
            offset: Optional position offset from object center [x, y, z]

        Returns:
            CameraConfig for egocentric camera
        """
        if initial_lookat is None:
            initial_lookat = np.array([0.0, 0.0, 0.25])

        # Egocentric uses very small distance to approximate attachment
        # Actual implementation will override position to object location + offset
        return CameraConfig(
            name=f"egocentric_{object_name}",
            type="tracking",
            lookat=initial_lookat,
            distance=0.01,  # Minimal distance (will be at object location)
            azimuth=0.0,
            elevation=0.0,  # Look forward from object
            fov=90.0,       # Wide FOV for immersive view
            track_object=object_name
        )


def get_camera_mode(
    mode: str,
    scene_center: Optional[np.ndarray] = None,
    track_object: Optional[str] = None
) -> List[CameraConfig]:
    """Get list of cameras for a named mode.

    Supported modes:
    - "front": Single fixed front camera
    - "multi": Three fixed cameras (front, side, top)
    - "tracking": Single camera tracking an object
    - "egocentric": First-person view from object

    Args:
        mode: Camera mode name
        scene_center: Optional scene center point for fixed cameras
        track_object: Required for "tracking" and "egocentric" modes

    Returns:
        List of CameraConfig objects for the mode

    Raises:
        ValueError: If mode is unknown or required parameters are missing

    Example:
        # Multi-camera setup
        cameras = get_camera_mode("multi", scene_center=np.array([0, 0, 0.3]))

        # Tracking setup
        cameras = get_camera_mode("tracking", track_object="sphere1")
    """
    if mode == "front":
        return [CameraPreset.fixed_front(scene_center)]

    elif mode == "multi":
        return [
            CameraPreset.fixed_front(scene_center),
            CameraPreset.fixed_side(scene_center),
            CameraPreset.fixed_top(scene_center)
        ]

    elif mode == "tracking":
        if track_object is None:
            raise ValueError("tracking mode requires track_object parameter")
        return [CameraPreset.tracking(track_object, scene_center)]

    elif mode == "egocentric":
        if track_object is None:
            raise ValueError("egocentric mode requires track_object parameter")
        return [CameraPreset.egocentric(track_object, scene_center)]

    else:
        raise ValueError(
            f"Unknown camera mode '{mode}'. "
            f"Supported: front, multi, tracking, egocentric"
        )
