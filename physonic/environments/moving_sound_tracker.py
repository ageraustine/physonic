"""Track and render moving sound sources in 3D space.

This module handles:
- Updating sound source positions from MuJoCo simulation
- Rendering spatial audio for moving sources
- Recording trajectories for analysis
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import mujoco

from physonic.synthesis.continuous import ContinuousSoundSpec


@dataclass
class MovingSoundSource:
    """A sound source that moves through 3D space.

    Attributes:
        sound_spec: The continuous sound specification
        body_name: Name of the MuJoCo body to track
        trajectory: List of (time, position) tuples
    """
    sound_spec: ContinuousSoundSpec
    body_name: str
    trajectory: List[Tuple[float, Tuple[float, float, float]]]

    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = []

    def update_position(self, time: float, position: Tuple[float, float, float]):
        """Update the sound source position.

        Args:
            time: Simulation time
            position: New (x, y, z) position
        """
        self.trajectory.append((time, position))
        self.sound_spec.position = position


class MovingSoundTracker:
    """Tracks multiple moving sound sources in a MuJoCo simulation.

    Usage:
        tracker = MovingSoundTracker(model, data)
        tracker.add_sound("robot_1", robot_sound_spec)

        # In simulation loop:
        for t in simulation_times:
            mujoco.mj_step(model, data)
            tracker.update(t)

        # Render audio
        audio = tracker.render_audio(duration=10.0)
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """Initialize tracker.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
        self.moving_sources: Dict[str, MovingSoundSource] = {}
        self.static_sources: List[ContinuousSoundSpec] = []

    def add_moving_sound(self, body_name: str, sound_spec: ContinuousSoundSpec):
        """Add a moving sound source attached to a body.

        Args:
            body_name: Name of MuJoCo body to track
            sound_spec: Sound specification
        """
        source = MovingSoundSource(
            sound_spec=sound_spec,
            body_name=body_name,
            trajectory=[],
        )
        self.moving_sources[body_name] = source

    def add_static_sound(self, sound_spec: ContinuousSoundSpec):
        """Add a static (non-moving) sound source.

        Args:
            sound_spec: Sound specification
        """
        self.static_sources.append(sound_spec)

    def update(self, time: float):
        """Update all moving sound positions from current simulation state.

        Args:
            time: Current simulation time
        """
        for body_name, source in self.moving_sources.items():
            # Get body ID
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                continue

            # Get body position
            pos = self.data.xpos[body_id].copy()
            source.update_position(time, tuple(pos))

    def get_sound_specs_at_time(self, time: float) -> List[ContinuousSoundSpec]:
        """Get all sound specifications interpolated to a specific time.

        Args:
            time: Time to interpolate to

        Returns:
            List of sound specs with positions at the given time
        """
        sounds = []

        # Add static sources
        sounds.extend(self.static_sources)

        # Add moving sources (interpolated)
        for source in self.moving_sources.values():
            if not source.trajectory:
                continue

            # Find position at this time (linear interpolation)
            if len(source.trajectory) == 1:
                # Only one position, use it
                _, pos = source.trajectory[0]
            else:
                # Interpolate between trajectory points
                times = [t for t, _ in source.trajectory]
                positions = np.array([p for _, p in source.trajectory])

                if time <= times[0]:
                    pos = tuple(positions[0])
                elif time >= times[-1]:
                    pos = tuple(positions[-1])
                else:
                    # Linear interpolation
                    pos = tuple(np.interp(time, times, positions.T))

            # Create sound spec with updated position
            sound_copy = ContinuousSoundSpec(
                source_id=source.sound_spec.source_id,
                sound_type=source.sound_spec.sound_type,
                fundamental_freq=source.sound_spec.fundamental_freq,
                harmonics=source.sound_spec.harmonics,
                noise_amount=source.sound_spec.noise_amount,
                modulation_freq=source.sound_spec.modulation_freq,
                modulation_depth=source.sound_spec.modulation_depth,
                position=pos,
                volume=source.sound_spec.volume,
            )
            sounds.append(sound_copy)

        return sounds

    def render_audio_trajectory(
        self,
        duration: float,
        listener_position: Tuple[float, float, float] = (0, 0, 1.5),
        hrtf_path: str = "assets/hrtf/test_kemar/",
        sample_rate: int = 44100,
    ) -> np.ndarray:
        """Render spatial audio with moving sounds.

        This renders the audio by:
        1. Generating continuous sounds for all sources
        2. Applying time-varying HRTF based on position changes
        3. Mixing all sources together

        Args:
            duration: Duration in seconds
            listener_position: Listener position (x, y, z)
            hrtf_path: Path to HRTF dataset
            sample_rate: Audio sample rate

        Returns:
            Binaural audio array (samples, 2)
        """
        from physonic.synthesis.continuous import synthesize_continuous_sound

        # For now, use simplified rendering at a single time point
        # TODO: Implement time-varying HRTF for moving sources
        sounds_at_mid = self.get_sound_specs_at_time(duration / 2)

        # Generate audio for each source and mix
        mixed = np.zeros(int(duration * sample_rate))

        for sound_spec in sounds_at_mid:
            audio = synthesize_continuous_sound(sound_spec, duration, sample_rate)
            mixed += audio * 0.5  # Simple mixing with attenuation

        # Normalize
        if np.max(np.abs(mixed)) > 0:
            mixed = mixed / np.max(np.abs(mixed)) * 0.8

        return mixed

    def get_trajectory_summary(self) -> Dict[str, List[Tuple[float, Tuple[float, float, float]]]]:
        """Get movement trajectories for all moving sources.

        Returns:
            Dictionary mapping body names to trajectories
        """
        return {
            body_name: source.trajectory
            for body_name, source in self.moving_sources.items()
        }


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Moving Sound Tracker Example")
    print("=" * 60)

    # Create simple scene
    from physonic.environments.dynamic_scene import generate_complex_data_center, build_complex_scene_xml

    dynamic_objects, moving_robots, continuous_sounds = generate_complex_data_center(
        room_size=(30, 25, 4),
        num_racks=24,
        num_robots=2,
        seed=42,
    )

    # Build MuJoCo model
    xml = build_complex_scene_xml(dynamic_objects, moving_robots, floor_size=(30, 25))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print(f"Model: {model.nbody} bodies, {model.ngeom} geoms")

    # Create tracker
    tracker = MovingSoundTracker(model, data)

    # Add sounds
    # Static sounds (racks, cooling)
    for sound in continuous_sounds:
        if "robot" not in sound.source_id:
            tracker.add_static_sound(sound)
        else:
            # Moving sounds (robots)
            body_name = sound.source_id
            tracker.add_moving_sound(body_name, sound)

    print(f"\nAdded {len(tracker.static_sources)} static sounds")
    print(f"Added {len(tracker.moving_sources)} moving sounds")

    # Simulate for 5 seconds
    print("\nSimulating robot movement...")
    dt = 0.01  # 10ms timesteps
    duration = 5.0
    num_steps = int(duration / dt)

    # Apply simple control to move robots
    for step in range(num_steps):
        t = step * dt

        # Simple control: robot_1 moves forward, robot_2 circles
        if model.nu > 0:  # Has actuators
            # Robot 1: move forward (x direction)
            data.ctrl[0] = 2.0  # x velocity
            data.ctrl[1] = 0.0  # y velocity
            data.ctrl[2] = 0.0  # rotation

            # Robot 2: circle
            if model.nu > 3:
                data.ctrl[3] = 1.0 * np.cos(t)  # x velocity
                data.ctrl[4] = 1.0 * np.sin(t)  # y velocity
                data.ctrl[5] = 1.0  # rotation

        # Step simulation
        mujoco.mj_step(model, data)

        # Update tracker every 100ms
        if step % 10 == 0:
            tracker.update(t)

    print(f"Simulation complete. Tracked {num_steps // 10} time points.")

    # Get trajectories
    trajectories = tracker.get_trajectory_summary()

    print("\nTrajectories:")
    for body_name, traj in trajectories.items():
        if len(traj) > 0:
            start_pos = traj[0][1]
            end_pos = traj[-1][1]
            distance = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
            print(f"  {body_name}: {len(traj)} points, moved {distance:.2f}m")

    # Plot trajectories
    fig, ax = plt.subplots(figsize=(10, 8))

    for body_name, traj in trajectories.items():
        if len(traj) == 0:
            continue

        times = [t for t, _ in traj]
        positions = np.array([p for _, p in traj])

        ax.plot(positions[:, 0], positions[:, 1], 'o-', label=body_name, markersize=3)
        ax.plot(positions[0, 0], positions[0, 1], 'g*', markersize=15)  # Start
        ax.plot(positions[-1, 0], positions[-1, 1], 'r*', markersize=15)  # End

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Robot Trajectories in Data Center")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.savefig("outputs/robot_trajectories.png", dpi=150)
    print(f"\nSaved trajectory plot to outputs/robot_trajectories.png")
