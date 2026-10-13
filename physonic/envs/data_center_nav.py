"""Data center navigation RL environment with acoustic perception.

The agent controls a robot navigating through a complex data center using:
- Mimi-compressed binaural audio observations
- Ultrasonic sensors for proximity detection
- Proprioceptive state (position, velocity)
- Continuous 2D velocity control

Compatible with Stable-Baselines3 PPO.
"""

import gymnasium as gym
import numpy as np
import mujoco
from typing import Tuple, Dict, Any, Optional
import os

from physonic.environments.dynamic_scene import (
    generate_complex_data_center,
    build_complex_scene_xml,
)
from physonic.environments.scene_optimizer import build_optimized_scene_xml
from physonic.envs.audio_observation import AudioObservationGenerator
from physonic.envs.ultrasonic_sensor import UltrasonicSensorArray


class DataCenterNavigationEnv(gym.Env):
    """Gymnasium environment for robot navigation in complex data center.

    **Task**: Navigate to randomly placed goal positions using acoustic cues and ultrasonic sensors.

    **Observations**:
        - Audio codes: (32, T) - Mimi-compressed binaural audio (T ≈ 6-8 frames for 0.5s)
        - Ultrasonic: (4,) - [front, left, right, back] distances in meters (0 to max_range)
        - Robot state: (6,) - [x, y, z, vx, vy, heading]

    **Actions**:
        - Continuous 2D velocity: (2,) - [vx, vy] in [-1, 1] m/s

    **Rewards**:
        - +10 for reaching goal (within 1m)
        - -0.01 * distance_to_goal (every step)
        - -5 for collision with obstacles
        - -10 for leaving bounds

    **Episode**:
        - Terminates when goal reached or max_steps exceeded
        - Max steps: 500 (50 seconds at 10 Hz)
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        room_size: Tuple[float, float, float] = (30, 25, 4),
        num_racks: int = 24,
        audio_duration: float = 0.5,  # 500ms audio windows
        control_freq: int = 10,  # 10 Hz control
        max_steps: int = 500,
        seed: Optional[int] = None,
        use_optimized_scene: bool = True,  # Enable scene optimization
        compound_group_size: int = 10,  # Objects per compound body
        # Audio observation settings
        use_mimi: bool = True,  # Enable Mimi codec compression
        max_audio_distance: float = 30.0,  # Spatial audio culling distance
        hrtf_path: Optional[str] = None,  # HRTF dataset path
        # Ultrasonic sensor settings
        ultrasonic_max_range: float = 5.0,  # Max ultrasonic range (meters)
        num_ultrasonic_sensors: int = 4,  # Number of ultrasonic sensors (4 or 8)
    ):
        """Initialize data center navigation environment.

        Args:
            room_size: (width, depth, height) in meters
            num_racks: Number of server racks
            audio_duration: Audio observation window duration (seconds)
            control_freq: Control frequency in Hz
            max_steps: Maximum steps per episode
            seed: Random seed
            use_optimized_scene: Enable MuJoCo optimization (compound bodies)
            compound_group_size: Number of objects per compound body (10-20 recommended)
            use_mimi: Enable Mimi codec compression for audio
            max_audio_distance: Maximum distance for spatial audio rendering (meters)
            hrtf_path: Path to HRTF dataset (defaults to assets/hrtf/test_kemar)
            ultrasonic_max_range: Maximum ultrasonic sensor range (meters)
            num_ultrasonic_sensors: Number of ultrasonic sensors (4 or 8)
        """
        super().__init__()

        self.room_size = room_size
        self.num_racks = num_racks
        self.audio_duration = audio_duration
        self.control_freq = control_freq
        self.max_steps = max_steps
        self.sample_rate = 44100
        self.use_optimized_scene = use_optimized_scene
        self.compound_group_size = compound_group_size
        self.use_mimi = use_mimi
        self.max_audio_distance = max_audio_distance
        self.ultrasonic_max_range = ultrasonic_max_range
        self.num_ultrasonic_sensors = num_ultrasonic_sensors

        # HRTF path (default to test dataset)
        if hrtf_path is None:
            # Assume project root structure
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            hrtf_path = os.path.join(project_root, "assets", "hrtf", "test_kemar")
        self.hrtf_path = hrtf_path

        # Observation space: Mimi codes + ultrasonic + robot state
        # Mimi codes shape: (32, time_frames) where time_frames ≈ audio_duration * 12.5
        # Note: Mimi uses 32 codebooks (not 8 as initially thought)
        mimi_time_frames = int(audio_duration * 12.5) + 2  # 12.5 Hz frame rate + margin

        self.observation_space = gym.spaces.Dict({
            "audio_codes": gym.spaces.Box(
                low=0, high=2047,  # Mimi codebook size
                shape=(32, mimi_time_frames),
                dtype=np.int64
            ),
            "ultrasonic": gym.spaces.Box(
                low=0.0, high=ultrasonic_max_range,
                shape=(num_ultrasonic_sensors,),
                dtype=np.float32
            ),
            "state": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(6,),  # x, y, z, vx, vy, heading
                dtype=np.float32
            ),
        })

        # Action space: continuous 2D velocity control
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.goal_position = None
        self.robot_start_position = None

        # RNG
        self._np_random = np.random.RandomState(seed)

        # Initialize MuJoCo simulation (this creates audio_generator and ultrasonic_sensors)
        self._init_mujoco()

    def _init_mujoco(self):
        """Initialize MuJoCo model and data."""
        # Generate complex data center
        dynamic_objects, moving_robots, continuous_sounds = generate_complex_data_center(
            room_size=self.room_size,
            num_racks=self.num_racks,
            num_robots=1,  # Just our controlled robot
            seed=0,
        )

        # Build XML (optimized or standard)
        if self.use_optimized_scene:
            # Use optimized scene builder (compound bodies, visual-only geoms)
            xml = build_optimized_scene_xml(
                dynamic_objects,
                moving_robots,
                floor_size=(self.room_size[0], self.room_size[1]),
                use_compound_bodies=True,
                compound_group_size=self.compound_group_size
            )
        else:
            # Use standard scene builder (individual bodies)
            xml = build_complex_scene_xml(
                dynamic_objects,
                moving_robots,
                floor_size=(self.room_size[0], self.room_size[1])
            )

        # Load model
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Get robot body ID
        self.robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot_1")

        # Initialize audio observation generator
        self.audio_generator = AudioObservationGenerator(
            continuous_sounds=continuous_sounds,
            hrtf_path=self.hrtf_path,
            audio_duration=self.audio_duration,
            sample_rate=self.sample_rate,
            max_audio_distance=self.max_audio_distance,
            use_mimi=self.use_mimi,
            mimi_parallel=False,  # Parallel is slower for short clips
        )

        # Initialize ultrasonic sensor array
        self.ultrasonic_sensors = UltrasonicSensorArray(
            model=self.model,
            robot_body_id=self.robot_body_id,
            max_range=self.ultrasonic_max_range,
            sensor_height=0.15,  # Robot height
            num_sensors=self.num_ultrasonic_sensors,
        )

    def _get_robot_state(self) -> np.ndarray:
        """Get current robot state.

        Returns:
            Array [x, y, z, vx, vy, heading]
        """
        pos = self.data.xpos[self.robot_body_id].copy()
        vel = self.data.cvel[self.robot_body_id][:3].copy()  # Linear velocity

        # Get heading from quaternion
        quat = self.data.xquat[self.robot_body_id]
        # Convert quaternion to heading angle
        heading = np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                            1 - 2 * (quat[2]**2 + quat[3]**2))

        return np.array([pos[0], pos[1], pos[2], vel[0], vel[1], heading], dtype=np.float32)

    def _generate_audio_observation(self) -> np.ndarray:
        """Generate audio observation from current state.

        Returns:
            Mimi codes array of shape (32, time_frames), zero-padded to match observation space
        """
        # Get robot position and orientation
        robot_pos = self.data.xpos[self.robot_body_id].copy()
        robot_quat = self.data.xquat[self.robot_body_id].copy()

        # Convert quaternion to heading angle (radians)
        robot_heading = np.arctan2(
            2 * (robot_quat[0] * robot_quat[3] + robot_quat[1] * robot_quat[2]),
            1 - 2 * (robot_quat[2]**2 + robot_quat[3]**2)
        )

        # Generate audio observation (Mimi codes)
        audio_codes = self.audio_generator.generate_observation(
            robot_position=robot_pos,
            robot_orientation=robot_heading,
        )

        # Pad to match observation space if needed
        expected_shape = self.observation_space["audio_codes"].shape
        if audio_codes.shape != expected_shape:
            padded = np.zeros(expected_shape, dtype=np.int64)
            # Copy actual codes into padded array
            actual_time_frames = min(audio_codes.shape[1], expected_shape[1])
            padded[:, :actual_time_frames] = audio_codes[:, :actual_time_frames]
            return padded

        return audio_codes

    def _generate_ultrasonic_observation(self) -> np.ndarray:
        """Generate ultrasonic sensor observation.

        Returns:
            Distance array of shape (num_sensors,)
        """
        return self.ultrasonic_sensors.read_distances(self.data)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment to initial state.

        Returns:
            observation: Dict with 'audio' and 'state'
            info: Dictionary with episode info
        """
        super().reset(seed=seed)

        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)

        # Set random robot start position (avoiding obstacles)
        self.robot_start_position = self._sample_free_position()
        self.data.qpos[0:2] = self.robot_start_position[:2]  # x, y

        # Set random goal position
        self.goal_position = self._sample_free_position()

        # Reset step counter
        self.current_step = 0

        # Get observation
        obs = {
            "audio_codes": self._generate_audio_observation(),
            "ultrasonic": self._generate_ultrasonic_observation(),
            "state": self._get_robot_state(),
        }

        info = {
            "goal_position": self.goal_position,
            "distance_to_goal": self._distance_to_goal(),
        }

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step.

        Args:
            action: [vx, vy] velocity commands in [-1, 1] m/s

        Returns:
            observation: Dict with 'audio' and 'state'
            reward: Scalar reward
            terminated: Whether episode ended successfully
            truncated: Whether episode was cut off
            info: Dictionary with step info
        """
        # Apply action (velocity control)
        vx, vy = action
        self.data.ctrl[0] = vx * 2.0  # Scale to reasonable velocity
        self.data.ctrl[1] = vy * 2.0

        # Step physics (multiple substeps for stability)
        steps_per_control = int(1.0 / (self.control_freq * self.model.opt.timestep))
        for _ in range(steps_per_control):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        # Get new state
        robot_state = self._get_robot_state()
        audio_codes = self._generate_audio_observation()
        ultrasonic = self._generate_ultrasonic_observation()

        obs = {
            "audio_codes": audio_codes,
            "ultrasonic": ultrasonic,
            "state": robot_state,
        }

        # Calculate reward
        distance = self._distance_to_goal()
        reward = -0.01 * distance  # Distance penalty

        # Check if goal reached
        terminated = False
        if distance < 1.0:  # Within 1 meter
            reward += 10.0
            terminated = True

        # Check for collisions (simple: check if robot is too low)
        if robot_state[2] < 0.1:  # Robot fell or collided
            reward -= 5.0
            terminated = True

        # Check bounds
        if not self._in_bounds(robot_state[:2]):
            reward -= 10.0
            terminated = True

        # Truncate if max steps
        truncated = self.current_step >= self.max_steps

        info = {
            "goal_position": self.goal_position,
            "distance_to_goal": distance,
            "is_success": distance < 1.0,
        }

        return obs, reward, terminated, truncated, info

    def _sample_free_position(self) -> np.ndarray:
        """Sample a random free position in the room.

        Returns:
            Position [x, y, z]
        """
        # Simple: sample random position with some margin from walls
        x = self._np_random.uniform(3, self.room_size[0] - 3)
        y = self._np_random.uniform(3, self.room_size[1] - 3)
        z = 0.15  # Robot height

        return np.array([x, y, z], dtype=np.float32)

    def _distance_to_goal(self) -> float:
        """Calculate Euclidean distance to goal.

        Returns:
            Distance in meters
        """
        robot_pos = self.data.xpos[self.robot_body_id]
        return np.linalg.norm(robot_pos[:2] - self.goal_position[:2])

    def _in_bounds(self, position: np.ndarray) -> bool:
        """Check if position is within room bounds.

        Args:
            position: [x, y]

        Returns:
            True if in bounds
        """
        x, y = position
        return (0 < x < self.room_size[0] and
                0 < y < self.room_size[1])

    def render(self):
        """Render environment (not implemented)."""
        pass

    def close(self):
        """Close environment."""
        pass


# Example usage
if __name__ == "__main__":
    print("Data Center Navigation Environment")
    print("=" * 60)

    # Create environment
    env = DataCenterNavigationEnv(
        room_size=(30, 25, 4),
        num_racks=24,
        use_mimi=True,
        max_steps=500,
        seed=42,
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test episode
    print("\nTesting random episode...")
    obs, info = env.reset()
    print(f"Initial state shape: {obs['state'].shape}")
    print(f"Audio codes shape: {obs['audio_codes'].shape}")
    print(f"Ultrasonic shape: {obs['ultrasonic'].shape}")
    print(f"Ultrasonic readings: {obs['ultrasonic']}")
    print(f"Goal position: {info['goal_position']}")

    total_reward = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 5 == 0:
            print(f"Step {step}: distance={info['distance_to_goal']:.2f}m, reward={reward:.3f}")

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    print(f"\nTotal reward: {total_reward:.3f}")
    print("Environment test complete!")
