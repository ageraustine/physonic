"""Binaural microphone positioning and angle computation for spatial audio."""
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class BinauralMicrophone:
    """Binaural microphone (simulated human head) in 3D space.

    Attributes:
        position: 3D position [x, y, z] in meters
        orientation: Yaw angle in degrees (0 = facing +X axis, 90 = facing +Y)

    Coordinate System:
        - +X: Forward
        - +Y: Left
        - +Z: Up
        - Right-handed coordinate system

    Example:
        >>> mic = BinauralMicrophone(position=np.array([0, 0, 1.5]), orientation=0)
        >>> mic.position
        array([0. , 0. , 1.5])
        >>> mic.orientation
        0.0
    """
    position: np.ndarray  # Shape (3,), [x, y, z] in meters
    orientation: float    # Yaw angle in degrees

    def compute_source_angle(self, source_position: np.ndarray) -> Tuple[float, float]:
        """Compute azimuth and elevation from microphone to source.

        The angles are computed in the microphone's local coordinate frame,
        accounting for the microphone's orientation.

        Args:
            source_position: Source 3D position [x, y, z] in meters

        Returns:
            (azimuth, elevation) in degrees:
                - azimuth: 0° = front, 90° = left, -90° = right, 180° = back
                - elevation: 0° = horizontal, positive = above, negative = below

        Example:
            >>> mic = BinauralMicrophone(position=np.array([0, 0, 1]), orientation=0)
            >>> # Source directly in front
            >>> azim, elev = mic.compute_source_angle(np.array([1, 0, 1]))
            >>> round(azim), round(elev)
            (0, 0)
            >>> # Source to the left
            >>> azim, elev = mic.compute_source_angle(np.array([0, 1, 1]))
            >>> round(azim), round(elev)
            (90, 0)
            >>> # Source above
            >>> azim, elev = mic.compute_source_angle(np.array([1, 0, 2]))
            >>> round(elev)
            45
        """
        # Compute relative vector from mic to source
        relative = source_position - self.position

        # Rotate by microphone orientation (yaw rotation around Z-axis)
        # To transform from world to mic frame, we rotate by -orientation
        theta = np.radians(-self.orientation)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Rotation matrix around Z-axis (yaw)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta,  cos_theta, 0],
            [0,          0,         1]
        ])

        # Transform to microphone's local frame
        relative_local = rotation_matrix @ relative

        # Compute azimuth: angle in XY plane
        # arctan2(y, x) gives angle from +X axis
        azimuth = np.degrees(np.arctan2(relative_local[1], relative_local[0]))

        # Compute elevation: angle from horizontal plane
        horizontal_distance = np.hypot(relative_local[0], relative_local[1])
        elevation = np.degrees(np.arctan2(relative_local[2], horizontal_distance))

        return float(azimuth), float(elevation)
