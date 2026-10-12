"""Video recording callback for PPO training.

Records evaluation episodes as MP4 videos to visualize agent behavior.
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import mujoco
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import cv2


class VideoRecorderCallback(BaseCallback):
    """Callback for recording training videos at evaluation intervals.

    Records episodes as MP4 videos showing the robot navigating the data center.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        video_folder: str = "outputs/training_videos",
        record_freq: int = 10000,  # Record every 10k steps
        video_length: int = 500,    # Max frames per video
        fps: int = 10,              # Video FPS
        name_prefix: str = "training",
        verbose: int = 0,
    ):
        """Initialize video recorder.

        Args:
            eval_env: Evaluation environment (unwrapped Gymnasium env)
            video_folder: Directory to save videos
            record_freq: Record video every N timesteps
            video_length: Maximum video length in frames
            fps: Video framerate
            name_prefix: Prefix for video filenames
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_folder = Path(video_folder)
        self.record_freq = record_freq
        self.video_length = video_length
        self.fps = fps
        self.name_prefix = name_prefix

        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.next_record_step = record_freq

    def _on_step(self) -> bool:
        """Check if we should record a video."""
        if self.num_timesteps >= self.next_record_step:
            self._record_video()
            self.next_record_step += self.record_freq
        return True

    def _record_video(self):
        """Record an evaluation episode as video."""
        video_path = self.video_folder / f"{self.name_prefix}_{self.num_timesteps:08d}_steps.mp4"

        if self.verbose > 0:
            print(f"\nRecording video: {video_path}")

        # Get the actual Gymnasium environment
        # Only works with DummyVecEnv (not SubprocVecEnv)
        try:
            # Check if this is DummyVecEnv
            if not hasattr(self.eval_env, 'envs'):
                if self.verbose > 0:
                    print("Warning: Video recording only works with DummyVecEnv (not SubprocVecEnv)")
                    print("  Use --n-envs 1 for evaluation environment to enable video recording")
                return

            # Unwrap to get the actual environment with MuJoCo
            env = self.eval_env.envs[0]  # Get first env from DummyVecEnv
            while hasattr(env, 'env'):
                env = env.env  # Unwrap Monitor, etc.

            # Check if environment has MuJoCo model
            if not hasattr(env, 'model') or not hasattr(env, 'data'):
                if self.verbose > 0:
                    print("Warning: Environment doesn't have MuJoCo model, skipping video")
                return

        except (AttributeError, IndexError) as e:
            if self.verbose > 0:
                print(f"Warning: Could not access environment: {e}")
                print("  Video recording requires DummyVecEnv, not SubprocVecEnv")
            return

        # Create MuJoCo renderer
        renderer = mujoco.Renderer(env.model, width=480, height=480)

        # Setup camera
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = 135
        camera.elevation = -30
        camera.distance = 35
        camera.lookat[:] = [15, 12.5, 1.0]  # Center of data center

        # Setup rendering options
        opt = mujoco.MjvOption()

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, self.fps, (480, 480))

        # Run episode
        obs, info = env.reset()
        frames = []
        total_reward = 0
        step_count = 0

        for step in range(self.video_length):
            # Render frame
            mujoco.mj_step(env.model, env.data)  # Update MuJoCo state
            mujoco.mjv_updateScene(
                env.model, env.data, opt, None, camera,
                mujoco.mjtCatBit.mjCAT_ALL, renderer.scene
            )
            pixels = renderer.render()

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

            # Add info overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame_bgr, f"Step: {step_count}", (10, 30),
                       font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Reward: {total_reward:.1f}", (10, 60),
                       font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Distance: {info.get('distance_to_goal', 0):.2f}m", (10, 90),
                       font, 0.7, (255, 255, 255), 2)

            # Draw goal marker (green circle)
            if hasattr(env, 'goal_position') and env.goal_position is not None:
                # Project 3D goal to 2D screen (simplified)
                # This is approximate - ideally we'd use proper projection matrix
                pass

            out.write(frame_bgr)

            # Take action
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        out.release()
        renderer.close()

        if self.verbose > 0:
            success = "SUCCESS" if info.get('is_success', False) else "FAILED"
            print(f"  Recorded {step_count} frames")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Result: {success}")
            print(f"  Saved to: {video_path}")


if __name__ == "__main__":
    # Test the video recorder callback
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from physonic.envs.data_center_nav import DataCenterNavigationEnv

    print("Testing VideoRecorderCallback")
    print("=" * 70)

    # Create environment
    def make_env():
        env = DataCenterNavigationEnv(seed=42)
        return env

    env = DummyVecEnv([make_env])

    # Create simple PPO model (untrained)
    model = PPO("MultiInputPolicy", env, verbose=1)

    # Create callback
    callback = VideoRecorderCallback(
        eval_env=env,
        video_folder="outputs/test_videos",
        record_freq=100,  # Record immediately for testing
        verbose=1
    )

    # Train for a few steps to trigger video recording
    print("\nTraining for 200 steps to trigger video recording...")
    model.learn(total_timesteps=200, callback=callback, progress_bar=False)

    print("\n" + "=" * 70)
    print("Video recording test complete!")
    print("Check outputs/test_videos/ for recorded video")
