"""Real-time training visualization callback.

Shows live rendering of training episodes and records training rollouts.
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import mujoco
from stable_baselines3.common.callbacks import BaseCallback
import cv2


class TrainingMonitorCallback(BaseCallback):
    """Monitor and visualize training in real-time.

    This callback:
    1. Records videos from TRAINING rollouts (not just eval)
    2. Shows live rendering window (optional)
    3. Saves frames at regular intervals
    """

    def __init__(
        self,
        video_folder: str = "outputs/training_monitor",
        record_freq: int = 10000,       # Record every 10k steps
        record_duration: int = 100,      # Record 100 frames (10 seconds @ 10Hz)
        show_window: bool = False,       # Show live window (SLOW!)
        fps: int = 10,
        verbose: int = 0,
    ):
        """Initialize training monitor.

        Args:
            video_folder: Directory to save videos
            record_freq: Start recording every N timesteps
            record_duration: Number of frames to record
            show_window: Show live OpenCV window (very slow, for debugging only)
            fps: Video framerate
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.video_folder = Path(video_folder)
        self.record_freq = record_freq
        self.record_duration = record_duration
        self.show_window = show_window
        self.fps = fps

        self.video_folder.mkdir(parents=True, exist_ok=True)

        # Recording state
        self.is_recording = False
        self.recording_frames = []
        self.current_video_path = None
        self.next_record_step = record_freq
        self.frames_recorded = 0

        # MuJoCo renderer (will be initialized on first step)
        self.renderer = None
        self.camera = None
        self.env_model = None
        self.env_data = None

    def _init_renderer(self):
        """Initialize MuJoCo renderer from training environment."""
        try:
            # Check if this is DummyVecEnv (required for video recording)
            if not hasattr(self.training_env, 'envs'):
                if self.verbose > 0:
                    print("Warning: Training monitor only works with DummyVecEnv (not SubprocVecEnv)")
                    print("  Use --n-envs 1 to enable training video recording")
                    print("  Or use --record-video instead for evaluation videos")
                return False

            # Get the actual environment from VecEnv
            env = self.training_env.envs[0]
            while hasattr(env, 'env'):
                env = env.env

            if not hasattr(env, 'model') or not hasattr(env, 'data'):
                if self.verbose > 0:
                    print("Warning: Environment doesn't have MuJoCo model")
                return False

            self.env_model = env.model
            self.env_data = env.data

            # Create renderer
            self.renderer = mujoco.Renderer(self.env_model, width=480, height=480)

            # Setup camera
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.azimuth = 135
            self.camera.elevation = -30
            self.camera.distance = 35
            self.camera.lookat[:] = [15, 12.5, 1.0]

            if self.verbose > 0:
                print(f"✓ Training monitor initialized")

            return True

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not initialize renderer: {e}")
            return False

    def _on_step(self) -> bool:
        """Called at each training step."""
        # Initialize renderer on first step
        if self.renderer is None:
            if not self._init_renderer():
                return True  # Continue training even if rendering fails

        # Check if we should start recording
        if not self.is_recording and self.num_timesteps >= self.next_record_step:
            self._start_recording()

        # Record frame if currently recording
        if self.is_recording:
            self._record_frame()

        return True

    def _start_recording(self):
        """Start recording a new video."""
        self.is_recording = True
        self.recording_frames = []
        self.frames_recorded = 0
        self.current_video_path = self.video_folder / f"training_rollout_{self.num_timesteps:08d}_steps.mp4"

        if self.verbose > 0:
            print(f"\n📹 Started recording training rollout at {self.num_timesteps:,} steps")

    def _record_frame(self):
        """Record a single frame from current training state."""
        if self.renderer is None or self.env_model is None:
            return

        try:
            # Render current state
            opt = mujoco.MjvOption()
            mujoco.mjv_updateScene(
                self.env_model, self.env_data, opt, None, self.camera,
                mujoco.mjtCatBit.mjCAT_ALL, self.renderer.scene
            )
            pixels = self.renderer.render()

            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

            # Add info overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame_bgr, f"Training Step: {self.num_timesteps:,}", (10, 30),
                       font, 0.6, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Frame: {self.frames_recorded}/{self.record_duration}", (10, 60),
                       font, 0.6, (0, 255, 0), 2)

            # Show window if requested
            if self.show_window:
                cv2.imshow('Training Monitor', frame_bgr)
                cv2.waitKey(1)

            # Store frame
            self.recording_frames.append(frame_bgr)
            self.frames_recorded += 1

            # Check if recording complete
            if self.frames_recorded >= self.record_duration:
                self._stop_recording()

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Frame recording failed: {e}")

    def _stop_recording(self):
        """Stop recording and save video."""
        if not self.recording_frames:
            self.is_recording = False
            return

        try:
            # Save video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(self.current_video_path), fourcc, self.fps, (480, 480))

            for frame in self.recording_frames:
                out.write(frame)

            out.release()

            if self.verbose > 0:
                print(f"✓ Saved training rollout: {self.current_video_path}")
                print(f"  Recorded {self.frames_recorded} frames")

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Failed to save video: {e}")

        finally:
            self.is_recording = False
            self.recording_frames = []
            self.next_record_step = self.num_timesteps + self.record_freq

    def _on_training_end(self):
        """Clean up on training end."""
        # Save any ongoing recording
        if self.is_recording:
            self._stop_recording()

        # Close window if open
        if self.show_window:
            cv2.destroyAllWindows()

        # Close renderer
        if self.renderer is not None:
            self.renderer.close()


class RealTimeViewer(BaseCallback):
    """Real-time interactive viewer for training (VERY SLOW - debugging only).

    Opens a live window showing what the agent is doing during training.
    WARNING: This significantly slows down training!
    """

    def __init__(
        self,
        update_freq: int = 10,  # Update display every 10 steps
        window_name: str = "PPO Training - Real-Time View",
        verbose: int = 0,
    ):
        """Initialize real-time viewer.

        Args:
            update_freq: Update display every N steps
            window_name: OpenCV window name
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.update_freq = update_freq
        self.window_name = window_name

        self.renderer = None
        self.camera = None
        self.env_model = None
        self.env_data = None
        self.step_count = 0

    def _init_renderer(self):
        """Initialize MuJoCo renderer."""
        try:
            # Check if this is DummyVecEnv
            if not hasattr(self.training_env, 'envs'):
                if self.verbose > 0:
                    print("Warning: Real-time viewer only works with DummyVecEnv (not SubprocVecEnv)")
                    print("  Use --n-envs 1 to enable real-time visualization")
                return False

            env = self.training_env.envs[0]
            while hasattr(env, 'env'):
                env = env.env

            if not hasattr(env, 'model') or not hasattr(env, 'data'):
                return False

            self.env_model = env.model
            self.env_data = env.data

            self.renderer = mujoco.Renderer(self.env_model, width=640, height=640)

            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.azimuth = 135
            self.camera.elevation = -30
            self.camera.distance = 35
            self.camera.lookat[:] = [15, 12.5, 1.0]

            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 640, 640)

            if self.verbose > 0:
                print(f"✓ Real-time viewer initialized")
                print(f"  Press 'q' to close window (training continues)")

            return True

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not initialize viewer: {e}")
            return False

    def _on_step(self) -> bool:
        """Update display."""
        if self.renderer is None:
            if not self._init_renderer():
                return True

        self.step_count += 1

        # Only update display periodically
        if self.step_count % self.update_freq != 0:
            return True

        try:
            # Render current state
            opt = mujoco.MjvOption()
            mujoco.mjv_updateScene(
                self.env_model, self.env_data, opt, None, self.camera,
                mujoco.mjtCatBit.mjCAT_ALL, self.renderer.scene
            )
            pixels = self.renderer.render()

            # Convert to BGR
            frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

            # Add overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame_bgr, f"Training Step: {self.num_timesteps:,}", (10, 30),
                       font, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Press 'q' to close", (10, frame_bgr.shape[0] - 20),
                       font, 0.6, (0, 255, 255), 2)

            # Display
            cv2.imshow(self.window_name, frame_bgr)

            # Check for key press (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if self.verbose > 0:
                    print("\nClosing viewer window (training continues)...")
                cv2.destroyWindow(self.window_name)
                self.renderer.close()
                self.renderer = None  # Don't update anymore

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Display update failed: {e}")

        return True

    def _on_training_end(self):
        """Clean up."""
        cv2.destroyAllWindows()
        if self.renderer is not None:
            self.renderer.close()


if __name__ == "__main__":
    # Test the training monitor
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from physonic.envs.data_center_nav import DataCenterNavigationEnv

    print("Testing TrainingMonitorCallback")
    print("=" * 70)

    # Create environment
    def make_env():
        return DataCenterNavigationEnv(seed=42)

    env = DummyVecEnv([make_env])

    # Create model
    model = PPO("MultiInputPolicy", env, verbose=0)

    # Create monitor callback
    monitor = TrainingMonitorCallback(
        video_folder="outputs/test_training_monitor",
        record_freq=500,      # Record every 500 steps (for testing)
        record_duration=50,   # Record 50 frames
        show_window=False,    # Don't show window for testing
        verbose=1
    )

    # Train
    print("\nTraining for 1500 steps (should record 3 videos)...")
    model.learn(total_timesteps=1500, callback=monitor, progress_bar=False)

    print("\n" + "=" * 70)
    print("Training monitor test complete!")
    print("Check outputs/test_training_monitor/ for recorded videos")
