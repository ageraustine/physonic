"""Episode recording utilities compatible with Gymnasium and HuggingFace.

Provides recording capabilities for RL episodes with audio, video, and metrics.
Compatible with Gymnasium's VideoRecorder API and HuggingFace dataset format.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import numpy as np
import soundfile as sf


@dataclass
class EpisodeRecording:
    """Recording of a single RL episode.

    Attributes:
        episode_id: Unique episode identifier
        observations: List of observations (one per step)
        actions: List of actions taken
        rewards: List of rewards received
        infos: List of info dicts
        audio: Optional audio array (if environment produces audio)
        video_path: Optional path to video file
        metadata: Additional episode metadata
    """
    episode_id: int
    observations: List[np.ndarray]
    actions: List[Any]
    rewards: List[float]
    infos: List[Dict[str, Any]]
    audio: Optional[np.ndarray] = None
    video_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EpisodeRecorder:
    """Records RL episodes with audio and optional video.

    Compatible with:
    - Gymnasium environments (follows VideoRecorder pattern)
    - HuggingFace datasets (can export to Arrow/Parquet)
    - Isaac Sim (preparation for future USD export)

    Example:
        >>> env = SoundLocalizationEnv()
        >>> recorder = EpisodeRecorder(output_dir="recordings")
        >>> recorder.start_episode(env)
        >>> obs, info = env.reset()
        >>> for step in range(100):
        ...     action = policy(obs)
        ...     recorder.record_step(obs, action, reward, info)
        ...     obs, reward, done, trunc, info = env.step(action)
        >>> recorder.save_episode()
    """

    def __init__(
        self,
        output_dir: str = "recordings",
        save_audio: bool = True,
        save_video: bool = False,
        save_observations: bool = True,
    ):
        """Initialize episode recorder.

        Args:
            output_dir: Directory to save recordings
            save_audio: Whether to save audio files
            save_video: Whether to save video files (requires video rendering)
            save_observations: Whether to save observation arrays
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_audio = save_audio
        self.save_video = save_video
        self.save_observations = save_observations

        # Current episode state
        self.episode_id: Optional[int] = None
        self.env = None
        self.observations: List[np.ndarray] = []
        self.actions: List[Any] = []
        self.rewards: List[float] = []
        self.infos: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

        # Episode counter
        self.num_episodes = 0

    def start_episode(self, env, metadata: Optional[Dict[str, Any]] = None):
        """Start recording a new episode.

        Args:
            env: Gymnasium environment
            metadata: Optional metadata dict (e.g., {"agent": "ppo", "seed": 42})
        """
        self.episode_id = self.num_episodes
        self.env = env
        self.observations = []
        self.actions = []
        self.rewards = []
        self.infos = []
        self.metadata = metadata or {}

        # Store environment info
        self.metadata.update({
            "env_name": env.__class__.__name__,
            "episode_id": self.episode_id,
        })

        if hasattr(env, "observation_space"):
            self.metadata["observation_shape"] = str(env.observation_space.shape)
        if hasattr(env, "action_space"):
            self.metadata["action_space"] = str(env.action_space)

    def record_step(
        self,
        observation: np.ndarray,
        action: Any,
        reward: float,
        info: Dict[str, Any],
    ):
        """Record a single step.

        Args:
            observation: Observation from environment
            action: Action taken
            reward: Reward received
            info: Info dict from environment
        """
        if self.episode_id is None:
            raise RuntimeError("Must call start_episode() before record_step()")

        if self.save_observations:
            self.observations.append(observation.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.infos.append(info.copy())

    def save_episode(self) -> str:
        """Save episode to disk.

        Returns:
            Path to saved episode directory
        """
        if self.episode_id is None:
            raise RuntimeError("No episode to save")

        # Create episode directory
        episode_dir = self.output_dir / f"episode_{self.episode_id:05d}"
        episode_dir.mkdir(exist_ok=True)

        # Save metadata
        metadata_path = episode_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        # Save actions and rewards (always save these)
        np.save(episode_dir / "actions.npy", np.array(self.actions))
        np.save(episode_dir / "rewards.npy", np.array(self.rewards))

        # Save infos as JSON
        with open(episode_dir / "infos.json", "w") as f:
            json.dump(self.infos, f, indent=2, default=str)

        # Save observations if requested
        if self.save_observations and self.observations:
            obs_array = np.stack(self.observations, axis=0)
            np.save(episode_dir / "observations.npy", obs_array)

        # Save audio if available
        if self.save_audio and self.infos:
            # Check if environment provides audio in info
            if "audio" in self.infos[0]:
                audio_data = self.infos[0]["audio"]
                audio_path = episode_dir / "audio.wav"
                sf.write(audio_path, audio_data, 44100)

        # Save summary stats
        summary = {
            "episode_id": self.episode_id,
            "num_steps": len(self.actions),
            "total_reward": float(np.sum(self.rewards)),
            "mean_reward": float(np.mean(self.rewards)),
            "success": self.infos[-1].get("success", None) if self.infos else None,
        }

        # Add environment-specific metrics
        if hasattr(self.env, "get_metrics"):
            summary["metrics"] = self.env.get_metrics()

        with open(episode_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Increment counter
        self.num_episodes += 1

        return str(episode_dir)

    def load_episode(self, episode_path: str) -> EpisodeRecording:
        """Load a previously recorded episode.

        Args:
            episode_path: Path to episode directory

        Returns:
            EpisodeRecording object
        """
        episode_dir = Path(episode_path)

        # Load metadata
        with open(episode_dir / "metadata.json") as f:
            metadata = json.load(f)

        # Load arrays
        actions = np.load(episode_dir / "actions.npy", allow_pickle=True)
        rewards = np.load(episode_dir / "rewards.npy")

        # Load infos
        with open(episode_dir / "infos.json") as f:
            infos = json.load(f)

        # Load observations if available
        observations = None
        obs_path = episode_dir / "observations.npy"
        if obs_path.exists():
            observations = np.load(obs_path)

        # Load audio if available
        audio = None
        audio_path = episode_dir / "audio.wav"
        if audio_path.exists():
            audio, _ = sf.read(audio_path)

        return EpisodeRecording(
            episode_id=metadata["episode_id"],
            observations=list(observations) if observations is not None else [],
            actions=list(actions),
            rewards=list(rewards),
            infos=infos,
            audio=audio,
            metadata=metadata,
        )

    def export_to_hf_format(self, output_path: str):
        """Export all recorded episodes to HuggingFace dataset format.

        This creates a format compatible with HuggingFace's datasets library
        for easy upload to the Hub.

        Args:
            output_path: Path to output directory for HF dataset
        """
        # TODO: Implement HuggingFace export
        # Will use datasets.Dataset.from_dict() with proper schema
        raise NotImplementedError("HuggingFace export coming soon")

    def export_to_isaac_format(self, output_path: str):
        """Export episode data to Isaac Sim compatible format (USD).

        Prepares data for future Isaac Sim integration.

        Args:
            output_path: Path to output USD file
        """
        # TODO: Implement USD export for Isaac Sim
        # Will convert MuJoCo scenes to USD format
        raise NotImplementedError("Isaac Sim USD export coming soon")


# Example usage
if __name__ == "__main__":
    from physonic.envs import SoundLocalizationEnv

    print("Episode Recorder Example")
    print("=" * 60)

    # Create environment and recorder
    env = SoundLocalizationEnv(
        audio_duration=0.5,
        use_mel_spectrogram=True,
        hrtf_path="assets/hrtf/test_kemar/",
    )

    recorder = EpisodeRecorder(
        output_dir="outputs/recordings",
        save_audio=True,
        save_observations=True,
    )

    # Record 3 episodes
    for episode_num in range(3):
        print(f"\nRecording episode {episode_num}...")

        recorder.start_episode(env, metadata={"agent": "random", "seed": episode_num})

        obs, info = env.reset(seed=episode_num)

        for step in range(10):
            # Random agent
            action = env.action_space.sample()

            recorder.record_step(obs, action, 0.0, info)

            obs, reward, done, trunc, info = env.step(action)
            recorder.record_step(obs, action, reward, info)

            if done or trunc:
                break

        episode_path = recorder.save_episode()
        print(f"  Saved to: {episode_path}")

    print(f"\nRecorded {recorder.num_episodes} episodes")
    print(f"Output directory: {recorder.output_dir}")

    # Load and display first episode
    print("\nLoading episode 0...")
    recording = recorder.load_episode("outputs/recordings/episode_00000")
    print(f"  Steps: {len(recording.actions)}")
    print(f"  Total reward: {sum(recording.rewards):.3f}")
    print(f"  Metadata: {recording.metadata}")
