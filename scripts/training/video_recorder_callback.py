"""Video recording callback for PPO training.

Records evaluation episodes as MP4 videos with binaural audio to visualize agent behavior.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, List
import numpy as np
import mujoco
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import cv2
import soundfile as sf

# Import Physonic audio synthesis and spatial rendering
from physonic.synthesis.continuous import synthesize_continuous_sound, ContinuousSoundSpec
from physonic.spatial.hrtf import load_kemar_hrtf, HRTFSet
from physonic.spatial.microphone import BinauralMicrophone
from physonic.spatial.renderer_enhanced import render_spatial_source_enhanced


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
        enable_audio: bool = True,  # Enable binaural audio rendering
        audio_sample_rate: int = 44100,
        hrtf_path: str = "assets/hrtf/test_kemar/",
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
            enable_audio: Enable binaural audio rendering
            audio_sample_rate: Audio sample rate (Hz)
            hrtf_path: Path to HRTF dataset
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_folder = Path(video_folder)
        self.record_freq = record_freq
        self.video_length = video_length
        self.fps = fps
        self.name_prefix = name_prefix
        self.enable_audio = enable_audio
        self.audio_sample_rate = audio_sample_rate

        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.next_record_step = record_freq

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

    def _on_step(self) -> bool:
        """Check if we should record a video."""
        if self.num_timesteps >= self.next_record_step:
            self._record_video()
            self.next_record_step += self.record_freq
        return True

    def _render_binaural_audio_frame(
        self,
        continuous_sounds: List[ContinuousSoundSpec],
        robot_position: np.ndarray,
        robot_heading: float,
        frame_duration: float,
    ) -> np.ndarray:
        """Render binaural audio for a single frame with complete acoustic physics.

        Args:
            continuous_sounds: List of continuous sound sources in the scene
            robot_position: Robot's 3D position [x, y, z]
            robot_heading: Robot's heading angle in radians
            frame_duration: Duration of this frame in seconds

        Returns:
            Binaural stereo audio for this frame, shape (num_samples, 2)
        """
        if not continuous_sounds or self.hrtf is None:
            # Return silence
            num_samples = int(frame_duration * self.audio_sample_rate)
            return np.zeros((num_samples, 2), dtype=np.float32)

        # Calculate robot velocity (for Doppler effect)
        robot_velocity = np.zeros(3)
        if self.previous_robot_position is not None:
            displacement = robot_position - self.previous_robot_position
            robot_velocity = displacement / frame_duration  # m/s
        self.previous_robot_position = robot_position.copy()

        # Create binaural microphone at robot position with actual heading
        # Robot's ears rotate with its body - crucial for binaural cues!
        microphone = BinauralMicrophone(
            position=robot_position,
            orientation=np.degrees(robot_heading),  # Convert radians to degrees
        )

        # Synthesize each continuous sound source
        num_samples = int(frame_duration * self.audio_sample_rate)

        # We need to account for propagation delay - max delay could extend the buffer
        max_distance = max([np.linalg.norm(np.array(s.position) - robot_position)
                           for s in continuous_sounds] + [0])
        max_delay_samples = int((max_distance / 343.0) * self.audio_sample_rate)
        buffer_length = num_samples + max_delay_samples

        binaural_mix = np.zeros((buffer_length, 2), dtype=np.float32)

        for sound_spec in continuous_sounds:
            # Generate mono audio for this source
            mono_audio = synthesize_continuous_sound(
                sound_spec,
                duration=frame_duration,
                sample_rate=self.audio_sample_rate,
            )

            # Apply complete acoustic physics (Doppler, air absorption, delay, HRTF)
            source_position = np.array(sound_spec.position)
            stereo_audio = render_spatial_source_enhanced(
                mono_audio,
                source_position,
                microphone,
                self.hrtf,
                receiver_velocity=robot_velocity,
                source_velocity=None,  # Sources are stationary
                apply_doppler=True,
                apply_absorption=True,
                apply_delay=True,
                occlusion_factor=0.0,  # No occlusion detection yet
                sample_rate=self.audio_sample_rate,
            )

            # Mix into binaural output (handle different lengths from delay)
            mix_length = min(len(binaural_mix), len(stereo_audio))
            binaural_mix[:mix_length] += stereo_audio[:mix_length]

        # Trim to expected length (remove delay padding)
        binaural_mix = binaural_mix[:num_samples]

        # Normalize to prevent clipping
        max_amplitude = np.abs(binaural_mix).max()
        if max_amplitude > 0:
            binaural_mix = binaural_mix / max_amplitude * 0.7  # Leave headroom

        return binaural_mix

    def _record_video(self):
        """Record an evaluation episode as video with binaural audio."""
        # File paths
        video_only_path = self.video_folder / f"{self.name_prefix}_{self.num_timesteps:08d}_steps_video_only.mp4"
        audio_path = self.video_folder / f"{self.name_prefix}_{self.num_timesteps:08d}_steps.wav"
        final_video_path = self.video_folder / f"{self.name_prefix}_{self.num_timesteps:08d}_steps.mp4"

        if self.verbose > 0:
            audio_status = "with binaural audio" if self.enable_audio else "without audio"
            print(f"\nRecording video {audio_status}: {final_video_path}")

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

        # Setup camera (close third-person view)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = 135
        camera.elevation = -35  # Less steep for better robot view
        camera.distance = 8      # Much closer (was 20)
        camera.lookat[:] = [15, 12.5, 0.6]  # Robot height

        # Setup rendering options
        opt = mujoco.MjvOption()

        # Initialize video writer (video only, audio added later)
        # Use 'avc1' (H.264) for macOS AVFoundation compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(video_only_path), fourcc, self.fps, (480, 480))

        # Initialize audio buffer
        audio_buffer = [] if self.enable_audio else None
        frame_duration = 1.0 / self.fps  # Duration per frame in seconds

        # Run episode
        obs, info = env.reset()
        frames = []
        total_reward = 0
        step_count = 0

        # Get robot body ID for camera following
        robot_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "robot_1")

        # Get continuous sounds from environment's audio generator
        continuous_sounds = []
        if self.enable_audio and hasattr(env, 'audio_generator') and hasattr(env.audio_generator, 'continuous_sounds'):
            continuous_sounds = env.audio_generator.continuous_sounds
            if self.verbose > 1:
                print(f"  Found {len(continuous_sounds)} continuous sound sources")

        # Reset velocity tracking for new episode
        self.previous_robot_position = None

        for step in range(self.video_length):
            # Update camera to follow robot
            robot_pos = env.data.xpos[robot_body_id].copy()
            camera.lookat[:] = robot_pos  # Camera follows robot

            # Get robot heading from quaternion
            quat = env.data.xquat[robot_body_id]
            robot_heading = np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                                       1 - 2 * (quat[2]**2 + quat[3]**2))

            # Render binaural audio for this frame (if enabled)
            # Audio rotates with robot heading - crucial for spatial perception!
            if self.enable_audio and continuous_sounds:
                frame_audio = self._render_binaural_audio_frame(
                    continuous_sounds,
                    robot_pos,
                    robot_heading,
                    frame_duration,
                )
                audio_buffer.append(frame_audio)

            # Render frame
            mujoco.mj_step(env.model, env.data)  # Update MuJoCo state
            mujoco.mjv_updateScene(
                env.model, env.data, opt, None, camera,
                mujoco.mjtCatBit.mjCAT_ALL, renderer.scene
            )
            pixels = renderer.render()

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

            # Add sound source indicators (ripples) using proper MuJoCo projection
            if hasattr(env, 'audio_generator') and hasattr(env.audio_generator, 'continuous_sounds'):
                for sound_spec in env.audio_generator.continuous_sounds:
                    # Find the body/geom for this sound source
                    try:
                        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, sound_spec.source_id)
                        sound_pos = env.data.xpos[body_id]
                    except:
                        # If body not found, skip
                        continue

                    # Project 3D position to 2D screen using simple perspective projection
                    # This is a rough approximation - for exact projection would need full camera matrix
                    cam_to_obj = sound_pos - camera.lookat

                    # Camera coordinate frame (simplified)
                    azim_rad = np.radians(camera.azimuth)
                    elev_rad = np.radians(camera.elevation)

                    # Transform to camera space (rough approximation)
                    dx = cam_to_obj[0] * np.cos(azim_rad) + cam_to_obj[1] * np.sin(azim_rad)
                    dy = -cam_to_obj[0] * np.sin(azim_rad) + cam_to_obj[1] * np.cos(azim_rad)
                    dz = cam_to_obj[2]

                    # Perspective projection
                    if camera.distance - dx > 0.1:  # Avoid division by zero
                        screen_scale = 480 / (2 * camera.distance)
                        screen_x = int(240 + dy * screen_scale * 15)
                        screen_y = int(240 - (dz - dx * np.sin(elev_rad)) * screen_scale * 15)

                        # Clamp to image bounds
                        if 30 < screen_x < 450 and 30 < screen_y < 450:
                            # Draw pulsing ripple (animated based on step count)
                            pulse = int(15 + 5 * np.sin(step * 0.3))

                            # Color by sound type
                            if 'server' in sound_spec.sound_type:
                                color = (255, 100, 100)  # Blue for servers
                            elif 'cooling' in sound_spec.sound_type:
                                color = (100, 200, 255)  # Orange for cooling
                            else:
                                color = (200, 200, 200)  # Gray default

                            # Draw concentric circles for ripple effect
                            cv2.circle(frame_bgr, (screen_x, screen_y), pulse, color, 1, cv2.LINE_AA)
                            cv2.circle(frame_bgr, (screen_x, screen_y), pulse + 5, color, 1, cv2.LINE_AA)
                            cv2.circle(frame_bgr, (screen_x, screen_y), 3, color, -1)  # Center dot

            # Add info overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame_bgr, f"Step: {step_count}", (10, 30),
                       font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Reward: {total_reward:.1f}", (10, 60),
                       font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Distance: {info.get('distance_to_goal', 0):.2f}m", (10, 90),
                       font, 0.7, (255, 255, 255), 2)

            # Add legend for sound indicators
            cv2.putText(frame_bgr, "Sound Sources:", (10, 430), font, 0.5, (255, 255, 255), 1)
            cv2.circle(frame_bgr, (20, 450), 5, (255, 100, 100), -1)
            cv2.putText(frame_bgr, "Servers", (30, 455), font, 0.5, (255, 255, 255), 1)
            cv2.circle(frame_bgr, (110, 450), 5, (100, 200, 255), -1)
            cv2.putText(frame_bgr, "Cooling", (120, 455), font, 0.5, (255, 255, 255), 1)

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

        # Combine audio and video
        if self.enable_audio and audio_buffer:
            try:
                # Concatenate all audio frames
                full_audio = np.concatenate(audio_buffer, axis=0)

                # Save audio as WAV
                sf.write(str(audio_path), full_audio, self.audio_sample_rate)

                if self.verbose > 1:
                    print(f"  Saved audio: {audio_path}")
                    print(f"    Audio duration: {len(full_audio) / self.audio_sample_rate:.2f}s")
                    print(f"    Audio samples: {len(full_audio)}")

                # Mux video + audio using ffmpeg
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_only_path),  # Video input
                    "-i", str(audio_path),       # Audio input
                    "-c:v", "copy",              # Copy video codec (no re-encoding)
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
                        print(f"  ✓ Combined video + audio: {final_video_path}")
                else:
                    if self.verbose > 0:
                        print(f"  Warning: ffmpeg muxing failed (code {result.returncode})")
                        print(f"  Video saved without audio: {video_only_path}")
                        if self.verbose > 1:
                            print(f"  ffmpeg error: {result.stderr}")

            except Exception as e:
                if self.verbose > 0:
                    print(f"  Warning: Audio processing failed: {e}")
                    print(f"  Video saved without audio: {video_only_path}")

        else:
            # No audio - rename video_only to final path
            if video_only_path.exists():
                video_only_path.rename(final_video_path)

        if self.verbose > 0:
            success = "SUCCESS" if info.get('is_success', False) else "FAILED"
            print(f"  Recorded {step_count} frames")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Result: {success}")
            print(f"  Saved to: {final_video_path}")


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
