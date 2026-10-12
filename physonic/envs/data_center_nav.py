"""Data center navigation RL environment with acoustic perception.

The agent controls a robot navigating through a complex data center using:
- Binaural audio observations (mel-spectrograms)
- Proprioceptive state (position, velocity)
- Continuous 2D velocity control

Compatible with Stable-Baselines3 PPO.
"""

import gymnasium as gym
import numpy as np
import mujoco
from typing import Tuple, Dict, Any, Optional

from physonic.environments.dynamic_scene import (
    generate_complex_data_center,
    build_complex_scene_xml,
)
from physonic.environments.moving_sound_tracker import MovingSoundTracker


class DataCenterNavigationEnv(gym.Env):
    """Gymnasium environment for robot navigation in complex data center.

    **Task**: Navigate to randomly placed goal positions using acoustic cues.

    **Observations**:
        - Mel-spectrogram: (64, 87, 2) - binaural audio
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
        use_mel_spectrogram: bool = True,
        n_mels: int = 64,
        max_steps: int = 500,
        seed: Optional[int] = None,
    ):
        """Initialize data center navigation environment.

        Args:
            room_size: (width, depth, height) in meters
            num_racks: Number of server racks
            audio_duration: Audio observation window duration
            control_freq: Control frequency in Hz
            use_mel_spectrogram: Use mel-spectrogram vs raw audio
            n_mels: Number of mel bands
            max_steps: Maximum steps per episode
            seed: Random seed
        """
        super().__init__()

        self.room_size = room_size
        self.num_racks = num_racks
        self.audio_duration = audio_duration
        self.control_freq = control_freq
        self.use_mel_spectrogram = use_mel_spectrogram
        self.n_mels = n_mels
        self.max_steps = max_steps
        self.sample_rate = 44100

        # Observation space: mel-spectrogram + robot state
        if use_mel_spectrogram:
            mel_shape = (n_mels, int(audio_duration * 100), 2)  # Approx time frames
            robot_state_shape = (6,)  # x, y, z, vx, vy, heading

            self.observation_space = gym.spaces.Dict({
                "audio": gym.spaces.Box(
                    low=-80.0, high=0.0,
                    shape=mel_shape,
                    dtype=np.float32
                ),
                "state": gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=robot_state_shape,
                    dtype=np.float32
                ),
            })
        else:
            # Raw audio observation
            audio_samples = int(audio_duration * self.sample_rate)
            self.observation_space = gym.spaces.Dict({
                "audio": gym.spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(audio_samples, 2),
                    dtype=np.float32
                ),
                "state": gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(6,),
                    dtype=np.float32
                ),
            })

        # Action space: continuous 2D velocity control
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Initialize MuJoCo simulation
        self._init_mujoco()

        # Initialize sound tracker
        self.sound_tracker = None

        # Episode state
        self.current_step = 0
        self.goal_position = None
        self.robot_start_position = None

        # RNG
        self._np_random = np.random.RandomState(seed)

    def _init_mujoco(self):
        """Initialize MuJoCo model and data."""
        # Generate complex data center
        dynamic_objects, moving_robots, continuous_sounds = generate_complex_data_center(
            room_size=self.room_size,
            num_racks=self.num_racks,
            num_robots=1,  # Just our controlled robot
            seed=0,
        )

        # Build XML
        xml = build_complex_scene_xml(dynamic_objects, moving_robots, floor_size=(self.room_size[0], self.room_size[1]))

        # Load model
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Get robot body ID
        self.robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot_1")

        # Store sound specs
        self.continuous_sounds = continuous_sounds

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
            Mel-spectrogram or raw audio array
        """
        # For now, generate silent audio as placeholder
        # TODO: Implement proper audio rendering with moving sound tracker
        if self.use_mel_spectrogram:
            # Return zeros of correct shape
            mel_shape = (self.n_mels, int(self.audio_duration * 100), 2)
            return np.zeros(mel_shape, dtype=np.float32)
        else:
            audio_samples = int(self.audio_duration * self.sample_rate)
            return np.zeros((audio_samples, 2), dtype=np.float32)

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
            "audio": self._generate_audio_observation(),
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
        audio_obs = self._generate_audio_observation()

        obs = {
            "audio": audio_obs,
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
        use_mel_spectrogram=True,
        max_steps=500,
        seed=42,
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test episode
    print("\nTesting random episode...")
    obs, info = env.reset()
    print(f"Initial state shape: {obs['state'].shape}")
    print(f"Audio shape: {obs['audio'].shape}")
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
