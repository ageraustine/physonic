"""Real-time training visualization callback.

Shows live rendering of training episodes and records training rollouts with binaural audio.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, List
import numpy as np
import mujoco
from stable_baselines3.common.callbacks import BaseCallback
import cv2
import soundfile as sf

# Import Physonic audio synthesis and spatial rendering
from physonic.synthesis.continuous import synthesize_continuous_sound, ContinuousSoundSpec
from physonic.spatial.hrtf import load_kemar_hrtf, HRTFSet
from physonic.spatial.microphone import BinauralMicrophone
from physonic.spatial.renderer_enhanced import render_spatial_source_enhanced


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
        enable_audio: bool = True,       # Enable binaural audio rendering
        audio_sample_rate: int = 44100,
        hrtf_path: str = "assets/hrtf/test_kemar/",
        max_audio_distance: float = 30.0,  # Maximum audible distance (meters) - culling optimization
        verbose: int = 0,
    ):
        """Initialize training monitor.

        Args:
            video_folder: Directory to save videos
            record_freq: Start recording every N timesteps
            record_duration: Number of frames to record
            show_window: Show live OpenCV window (very slow, for debugging only)
            fps: Video framerate
            enable_audio: Enable binaural audio rendering
            audio_sample_rate: Audio sample rate (Hz)
            hrtf_path: Path to HRTF dataset
            max_audio_distance: Maximum distance for audible sounds (meters). Sounds beyond this distance are culled (not rendered).
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.video_folder = Path(video_folder)
        self.record_freq = record_freq
        self.record_duration = record_duration
        self.show_window = show_window
        self.fps = fps
        self.enable_audio = enable_audio
        self.audio_sample_rate = audio_sample_rate
        self.max_audio_distance = max_audio_distance

        self.video_folder.mkdir(parents=True, exist_ok=True)

        # Recording state
        self.is_recording = False
        self.recording_frames = []
        self.recording_audio = []
        self.current_video_path = None
        self.next_record_step = record_freq
        self.frames_recorded = 0
        self.continuous_sounds = []

        # MuJoCo renderer (will be initialized on first step)
        self.renderer = None
        self.camera = None
        self.env_model = None
        self.env_data = None

        # Load HRTF dataset for spatial audio
        self.hrtf = None
        if self.enable_audio:
            try:
                self.hrtf = load_kemar_hrtf(hrtf_path)
                if self.verbose > 0:
                    print(f"✓ Loaded HRTF dataset from {hrtf_path}")
                    print(f"✓ Acoustic physics enabled: Doppler, air absorption, propagation delay")
            except Exception as e:
                if self.verbose > 0:
                    print(f"Warning: Could not load HRTF dataset: {e}")
                    print("  Videos will be recorded without audio")
                self.enable_audio = False

        # Robot velocity tracking (for Doppler effect)
        self.previous_robot_position = None

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

            # Setup camera (close third-person view)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.azimuth = 135
            self.camera.elevation = -35  # Less steep for better robot view
            self.camera.distance = 8      # Much closer (was 20)
            self.camera.lookat[:] = [15, 12.5, 0.6]  # Robot height

            if self.verbose > 0:
                print(f"✓ Training monitor initialized")

            return True

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not initialize renderer: {e}")
            return False

    def _render_binaural_audio_frame(
        self,
        robot_position: np.ndarray,
        frame_duration: float,
    ) -> np.ndarray:
        """Render binaural audio for a single frame with complete acoustic physics.

        Args:
            robot_position: Robot's 3D position [x, y, z]
            frame_duration: Duration of this frame in seconds

        Returns:
            Binaural stereo audio for this frame, shape (num_samples, 2)
        """
        if not self.continuous_sounds or self.hrtf is None:
            # Return silence
            num_samples = int(frame_duration * self.audio_sample_rate)
            return np.zeros((num_samples, 2), dtype=np.float32)

        # Calculate robot velocity (for Doppler effect)
        robot_velocity = np.zeros(3)
        if self.previous_robot_position is not None:
            displacement = robot_position - self.previous_robot_position
            robot_velocity = displacement / frame_duration  # m/s
        self.previous_robot_position = robot_position.copy()

        # Create binaural microphone at robot position (robot "ears")
        microphone = BinauralMicrophone(
            position=robot_position,
            orientation=0.0,  # Facing forward along X axis
        )

        # Synthesize each continuous sound source
        num_samples = int(frame_duration * self.audio_sample_rate)

        # Account for propagation delay (only for audible range)
        max_delay_samples = int((self.max_audio_distance / 343.0) * self.audio_sample_rate)
        buffer_length = num_samples + max_delay_samples

        binaural_mix = np.zeros((buffer_length, 2), dtype=np.float32)

        # OPTIMIZATION: Spatial audio culling - only render sounds within audible range
        # Filter sound sources by distance to robot (perceptual culling)
        audible_sounds = []
        for sound_spec in self.continuous_sounds:
            source_position = np.array(sound_spec.position)
            distance = np.linalg.norm(source_position - robot_position)

            if distance <= self.max_audio_distance:
                audible_sounds.append((sound_spec, distance))

        # Sort by distance (render closest sounds first for better mixing)
        audible_sounds.sort(key=lambda x: x[1])

        if self.verbose > 1 and audible_sounds:
            print(f"  Rendering {len(audible_sounds)}/{len(self.continuous_sounds)} audible sources (within {self.max_audio_distance}m)")

        for sound_spec, distance in audible_sounds:
            # Generate mono audio for this source
            mono_audio = synthesize_continuous_sound(
                sound_spec,
                duration=frame_duration,
                sample_rate=self.audio_sample_rate,
            )

            # Apply complete acoustic physics
            source_position = np.array(sound_spec.position)
            stereo_audio = render_spatial_source_enhanced(
                mono_audio,
                source_position,
                microphone,
                self.hrtf,
                receiver_velocity=robot_velocity,
                source_velocity=None,
                apply_doppler=True,
                apply_absorption=True,
                apply_delay=True,
                occlusion_factor=0.0,
                sample_rate=self.audio_sample_rate,
            )

            # Mix into binaural output
            mix_length = min(len(binaural_mix), len(stereo_audio))
            binaural_mix[:mix_length] += stereo_audio[:mix_length]

        # Trim to expected length
        binaural_mix = binaural_mix[:num_samples]

        # Normalize to prevent clipping
        max_amplitude = np.abs(binaural_mix).max()
        if max_amplitude > 0:
            binaural_mix = binaural_mix / max_amplitude * 0.7  # Leave headroom

        return binaural_mix

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
        self.recording_audio = []
        self.frames_recorded = 0
        self.current_video_path = self.video_folder / f"training_rollout_{self.num_timesteps:08d}_steps.mp4"

        # Get continuous sounds from environment (if available)
        self.continuous_sounds = []
        if self.enable_audio and hasattr(self.training_env, 'envs'):
            try:
                env = self.training_env.envs[0]
                while hasattr(env, 'env'):
                    env = env.env
                if hasattr(env, 'continuous_sounds'):
                    self.continuous_sounds = env.continuous_sounds
                    if self.verbose > 1:
                        print(f"  Found {len(self.continuous_sounds)} continuous sound sources")
            except:
                pass

        # Reset velocity tracking for new recording
        self.previous_robot_position = None

        if self.verbose > 0:
            audio_status = "with binaural audio" if (self.enable_audio and self.continuous_sounds) else ""
            print(f"\n📹 Started recording training rollout at {self.num_timesteps:,} steps {audio_status}")

    def _record_frame(self):
        """Record a single frame from current training state."""
        if self.renderer is None or self.env_model is None:
            return

        try:
            # Update camera to follow robot
            robot_body_id = mujoco.mj_name2id(self.env_model, mujoco.mjtObj.mjOBJ_BODY, "robot_1")
            robot_pos = self.env_data.xpos[robot_body_id].copy()
            self.camera.lookat[:] = robot_pos

            # Render binaural audio for this frame (if enabled)
            if self.enable_audio and self.continuous_sounds:
                frame_duration = 1.0 / self.fps
                frame_audio = self._render_binaural_audio_frame(
                    robot_pos,
                    frame_duration,
                )
                self.recording_audio.append(frame_audio)

            # Render current state
            opt = mujoco.MjvOption()
            mujoco.mjv_updateScene(
                self.env_model, self.env_data, opt, None, self.camera,
                mujoco.mjtCatBit.mjCAT_ALL, self.renderer.scene
            )
            pixels = self.renderer.render()

            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

            # Add sound source indicators using actual body positions
            if hasattr(self.training_env, 'envs'):
                env = self.training_env.envs[0]
                while hasattr(env, 'env'):
                    env = env.env

                if hasattr(env, 'continuous_sounds'):
                    for sound_spec in env.continuous_sounds:
                        try:
                            body_id = mujoco.mj_name2id(self.env_model, mujoco.mjtObj.mjOBJ_BODY, sound_spec.source_id)
                            sound_pos = self.env_data.xpos[body_id]

                            # Project to screen using camera
                            cam_to_obj = sound_pos - self.camera.lookat
                            azim_rad = np.radians(self.camera.azimuth)
                            elev_rad = np.radians(self.camera.elevation)

                            dx = cam_to_obj[0] * np.cos(azim_rad) + cam_to_obj[1] * np.sin(azim_rad)
                            dy = -cam_to_obj[0] * np.sin(azim_rad) + cam_to_obj[1] * np.cos(azim_rad)
                            dz = cam_to_obj[2]

                            if self.camera.distance - dx > 0.1:
                                screen_scale = 480 / (2 * self.camera.distance)
                                screen_x = int(240 + dy * screen_scale * 15)
                                screen_y = int(240 - (dz - dx * np.sin(elev_rad)) * screen_scale * 15)

                                if 30 < screen_x < 450 and 30 < screen_y < 450:
                                    pulse = int(15 + 5 * np.sin(self.frames_recorded * 0.3))
                                    if 'server' in sound_spec.sound_type:
                                        color = (255, 100, 100)
                                    elif 'cooling' in sound_spec.sound_type:
                                        color = (100, 200, 255)
                                    else:
                                        color = (200, 200, 200)
                                    cv2.circle(frame_bgr, (screen_x, screen_y), pulse, color, 1, cv2.LINE_AA)
                                    cv2.circle(frame_bgr, (screen_x, screen_y), pulse + 5, color, 1, cv2.LINE_AA)
                                    cv2.circle(frame_bgr, (screen_x, screen_y), 3, color, -1)
                        except:
                            continue

            # Add info overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame_bgr, f"Training Step: {self.num_timesteps:,}", (10, 30),
                       font, 0.6, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Frame: {self.frames_recorded}/{self.record_duration}", (10, 60),
                       font, 0.6, (0, 255, 0), 2)

            # Add legend
            cv2.putText(frame_bgr, "Sound Sources:", (10, 430), font, 0.5, (255, 255, 255), 1)
            cv2.circle(frame_bgr, (20, 450), 5, (255, 100, 100), -1)
            cv2.putText(frame_bgr, "Servers", (30, 455), font, 0.5, (255, 255, 255), 1)
            cv2.circle(frame_bgr, (110, 450), 5, (100, 200, 255), -1)
            cv2.putText(frame_bgr, "Cooling", (120, 455), font, 0.5, (255, 255, 255), 1)

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
        """Stop recording and save video with audio."""
        if not self.recording_frames:
            self.is_recording = False
            return

        try:
            # File paths
            video_only_path = self.current_video_path.with_name(
                self.current_video_path.stem + "_video_only.mp4"
            )
            audio_path = self.current_video_path.with_suffix('.wav')
            final_video_path = self.current_video_path

            # Save video frames
            # Use 'avc1' (H.264) for macOS AVFoundation compatibility
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(str(video_only_path), fourcc, self.fps, (480, 480))

            for frame in self.recording_frames:
                out.write(frame)

            out.release()

            # Combine audio and video
            if self.enable_audio and self.recording_audio:
                try:
                    # Concatenate all audio frames
                    full_audio = np.concatenate(self.recording_audio, axis=0)

                    # Save audio as WAV
                    sf.write(str(audio_path), full_audio, self.audio_sample_rate)

                    if self.verbose > 1:
                        print(f"  Saved audio: {len(full_audio) / self.audio_sample_rate:.2f}s")

                    # Mux video + audio using ffmpeg
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", str(video_only_path),  # Video input
                        "-i", str(audio_path),       # Audio input
                        "-c:v", "copy",              # Copy video codec
                        "-c:a", "aac",               # AAC audio codec
                        "-b:a", "192k",              # Audio bitrate
                        "-shortest",                 # Match shortest stream
                        str(final_video_path)
                    ]

                    result = subprocess.run(
                        ffmpeg_cmd,
                        capture_output=True,
                        text=True,
                        check=False
                    )

                    if result.returncode == 0:
                        # Success - remove temporary files
                        video_only_path.unlink()
                        audio_path.unlink()

                        if self.verbose > 0:
                            print(f"✓ Saved training rollout with audio: {final_video_path}")
                            print(f"  Recorded {self.frames_recorded} frames")
                    else:
                        if self.verbose > 0:
                            print(f"Warning: ffmpeg muxing failed")
                            print(f"✓ Saved training rollout (video only): {video_only_path}")
                            print(f"  Recorded {self.frames_recorded} frames")

                except Exception as e:
                    if self.verbose > 0:
                        print(f"Warning: Audio processing failed: {e}")
                        print(f"✓ Saved training rollout (video only): {video_only_path}")
                        print(f"  Recorded {self.frames_recorded} frames")

            else:
                # No audio - rename video_only to final path
                if video_only_path.exists():
                    video_only_path.rename(final_video_path)

                if self.verbose > 0:
                    print(f"✓ Saved training rollout: {final_video_path}")
                    print(f"  Recorded {self.frames_recorded} frames")

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Failed to save video: {e}")

        finally:
            self.is_recording = False
            self.recording_frames = []
            self.recording_audio = []
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

            # Setup camera (close third-person view)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.azimuth = 135
            self.camera.elevation = -35  # Less steep for better robot view
            self.camera.distance = 8      # Much closer (was 20)
            self.camera.lookat[:] = [15, 12.5, 0.6]  # Robot height

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
