"""Ultrasonic sensor model for robot navigation.

Provides efficient distance measurements using MuJoCo raycasting.
"""

import numpy as np
import mujoco
from typing import List, Tuple


class UltrasonicSensorArray:
    """Ultrasonic sensor array for obstacle detection.

    Simulates a 4-sensor array (front, left, right, back) using MuJoCo raycasting.
    Efficient and suitable for RL training.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        robot_body_id: int,
        max_range: float = 5.0,
        sensor_height: float = 0.15,
        num_sensors: int = 4,
    ):
        """Initialize ultrasonic sensor array.

        Args:
            model: MuJoCo model
            robot_body_id: Robot body ID for position/orientation
            max_range: Maximum sensing range in meters
            sensor_height: Height offset from robot center
            num_sensors: Number of sensors (default: 4 for front/left/right/back)
        """
        self.model = model
        self.robot_body_id = robot_body_id
        self.max_range = max_range
        self.sensor_height = sensor_height
        self.num_sensors = num_sensors

        # Sensor directions (radians offset from robot heading)
        # Front, Left, Right, Back
        if num_sensors == 4:
            self.sensor_angles = [0, np.pi/2, -np.pi/2, np.pi]
        elif num_sensors == 8:
            # 8-way sensing for finer resolution
            self.sensor_angles = [
                0, np.pi/4, np.pi/2, 3*np.pi/4,
                np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4
            ]
        else:
            raise ValueError(f"num_sensors must be 4 or 8, got {num_sensors}")

    def read_distances(self, data: mujoco.MjData) -> np.ndarray:
        """Read distance measurements from all sensors.

        Args:
            data: MuJoCo data

        Returns:
            Distance array of shape (num_sensors,) in meters.
            max_range indicates no obstacle detected.
        """
        distances = np.full(self.num_sensors, self.max_range, dtype=np.float32)

        # Get robot position and orientation
        robot_pos = data.xpos[self.robot_body_id].copy()
        robot_quat = data.xquat[self.robot_body_id].copy()

        # Convert quaternion to heading angle (yaw)
        robot_heading = self._quat_to_yaw(robot_quat)

        # Raycast from each sensor
        for i, sensor_angle in enumerate(self.sensor_angles):
            # Compute ray direction in world frame
            direction_angle = robot_heading + sensor_angle
            direction = np.array([
                np.cos(direction_angle),
                np.sin(direction_angle),
                0.0  # Horizontal rays
            ])

            # Ray origin at robot position (with height offset)
            origin = robot_pos + np.array([0, 0, self.sensor_height])

            # Raycast
            distance = self._raycast(data, origin, direction)
            distances[i] = distance

        return distances

    def _raycast(
        self,
        data: mujoco.MjData,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> float:
        """Perform raycast and return distance to nearest obstacle.

        Args:
            data: MuJoCo data
            origin: Ray origin [x, y, z]
            direction: Ray direction (normalized)

        Returns:
            Distance to nearest obstacle, or max_range if none found
        """
        # Normalize direction
        direction = direction / np.linalg.norm(direction)

        # Simple ray-AABB (axis-aligned bounding box) intersection for all geoms
        min_distance = self.max_range

        for geom_id in range(self.model.ngeom):
            # Skip robot's own geom (geom 0 is usually the robot)
            if geom_id == 0:
                continue

            # Get geom position and size
            geom_pos = data.geom_xpos[geom_id]
            geom_size = self.model.geom_size[geom_id]

            # Simple AABB approximation (works for boxes and spheres)
            # Create bounding box around geom
            half_extent = np.array([geom_size[0], geom_size[0], geom_size[2]])
            bbox_min = geom_pos - half_extent
            bbox_max = geom_pos + half_extent

            # Ray-AABB intersection
            hit_dist = self._ray_aabb_intersection(origin, direction, bbox_min, bbox_max)

            if hit_dist is not None and 0 < hit_dist < min_distance:
                min_distance = hit_dist

        return min_distance

    def _ray_aabb_intersection(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
    ) -> float:
        """Compute ray-AABB intersection distance.

        Args:
            origin: Ray origin
            direction: Ray direction (normalized)
            bbox_min: AABB minimum corner
            bbox_max: AABB maximum corner

        Returns:
            Distance to intersection, or None if no hit
        """
        # Avoid division by zero - use safe division to suppress warnings
        safe_dir = np.where(np.abs(direction) < 1e-8, 1.0, direction)
        inv_dir = np.where(np.abs(direction) < 1e-8, 1e8, 1.0 / safe_dir)

        t1 = (bbox_min - origin) * inv_dir
        t2 = (bbox_max - origin) * inv_dir

        tmin = np.minimum(t1, t2)
        tmax = np.maximum(t1, t2)

        tmin_max = np.max(tmin)
        tmax_min = np.min(tmax)

        if tmin_max > tmax_min or tmax_min < 0:
            return None  # No intersection

        # Return nearest intersection distance
        return tmin_max if tmin_max > 0 else tmax_min

    def _ray_sphere_intersection(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        sphere_center: np.ndarray,
        sphere_radius: float,
    ) -> float:
        """Compute ray-sphere intersection distance.

        Args:
            origin: Ray origin
            direction: Ray direction (normalized)
            sphere_center: Sphere center
            sphere_radius: Sphere radius

        Returns:
            Distance to intersection, or None if no hit
        """
        # Vector from ray origin to sphere center
        oc = origin - sphere_center

        # Quadratic coefficients for ray-sphere intersection
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - sphere_radius**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return None  # No intersection

        # Compute nearest intersection distance
        t = (-b - np.sqrt(discriminant)) / (2*a)

        if t < 0:
            return None  # Intersection behind ray

        return t

    def _quat_to_yaw(self, quat: np.ndarray) -> float:
        """Convert quaternion to yaw angle (heading).

        Args:
            quat: Quaternion [w, x, y, z]

        Returns:
            Yaw angle in radians
        """
        w, x, y, z = quat
        yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y**2 + z**2))
        return yaw

    def visualize_rays(
        self,
        data: mujoco.MjData,
        distances: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate ray visualization data for rendering.

        Args:
            data: MuJoCo data
            distances: Distance measurements from read_distances()

        Returns:
            List of (origin, endpoint) tuples for each ray
        """
        robot_pos = data.xpos[self.robot_body_id].copy()
        robot_quat = data.xquat[self.robot_body_id].copy()
        robot_heading = self._quat_to_yaw(robot_quat)

        rays = []

        for i, sensor_angle in enumerate(self.sensor_angles):
            direction_angle = robot_heading + sensor_angle
            direction = np.array([
                np.cos(direction_angle),
                np.sin(direction_angle),
                0.0
            ])

            origin = robot_pos + np.array([0, 0, self.sensor_height])
            endpoint = origin + direction * distances[i]

            rays.append((origin, endpoint))

        return rays
