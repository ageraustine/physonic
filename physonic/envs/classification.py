"""Sound material classification RL environment."""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional

from physonic.spatial import load_kemar_hrtf, BinauralMicrophone, render_spatial_source
from physonic.contact.rig import SceneSpec, simulate, extract_events
from physonic.contact.modal_synth import sample_modes, render_impact, limit
from physonic.contact.materials import MATERIALS
from physonic.envs.preprocessing import extract_mel_spectrogram
from physonic.envs.rewards import classification_reward


class SoundClassificationEnv(gym.Env):
    """
    Sound material classification from binaural audio.

    Task: Agent hears an impact sound and must classify the material.

    Observation:
        - Mel-spectrogram (n_mels, time_frames, 2) for each ear
        - OR: Raw binaural audio (T, 2) stereo @ 44.1kHz

    Action:
        - Discrete(4): {0: steel, 1: glass, 2: wood, 3: rubber}

    Reward:
        - +1.0 for correct classification
        - 0.0 for incorrect classification

    Episode:
        - Single-step prediction task
        - Each episode: random material → audio → agent predicts → evaluate
    """

    metadata = {"render_modes": ["none"]}

    # Material to class index mapping
    MATERIAL_CLASSES = ["steel", "glass", "wood", "rubber"]

    def __init__(
        self,
        audio_duration: float = 1.0,
        sample_rate: int = 44100,
        use_mel_spectrogram: bool = True,
        n_mels: int = 64,
        scenario_type: str = "drop",
        surface: str = "glass",
        randomize_surface: bool = False,
        randomize_position: bool = True,
        room_acoustics: bool = False,
        hrtf_path: str = "assets/hrtf/mit_kemar/",
        seed: Optional[int] = None
    ):
        super().__init__()
        self.audio_duration = audio_duration
        self.sr = sample_rate
        self.use_mel = use_mel_spectrogram
        self.n_mels = n_mels
        self.scenario_type = scenario_type
        self.surface = surface
        self.randomize_surface = randomize_surface
        self.randomize_position = randomize_position
        self.room_acoustics = room_acoustics

        # Load HRTF once (expensive, do in __init__)
        self.hrtf = load_kemar_hrtf(hrtf_path)

        # Microphone at origin, facing forward
        self.mic = BinauralMicrophone(
            position=np.array([0.0, 0.0, 1.5]),  # 1.5m height
            orientation=0.0  # Facing +X
        )

        # RNG for reproducibility
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # Observation space (same as localization)
        if use_mel_spectrogram:
            time_frames = int(np.ceil(self.audio_duration * self.sr / 512))
            self.observation_space = gym.spaces.Box(
                low=-80.0,
                high=40.0,
                shape=(self.n_mels, time_frames, 2),
                dtype=np.float32
            )
        else:
            audio_samples = int(self.audio_duration * self.sr)
            self.observation_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(audio_samples, 2),
                dtype=np.float32
            )

        # Action space: discrete material classification
        self.action_space = gym.spaces.Discrete(len(self.MATERIAL_CLASSES))

        # Episode state
        self.true_material_idx = None
        self.true_material = None
        self.current_audio = None
        self.source_pos = None

        # Tracking for confusion matrix
        self.confusion_counts = np.zeros((4, 4), dtype=int)  # [true, pred]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate new audio with random material."""
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 1. Randomly select material to classify
        self.true_material_idx = int(self.rng.integers(0, len(self.MATERIAL_CLASSES)))
        self.true_material = self.MATERIAL_CLASSES[self.true_material_idx]

        # 2. Optionally randomize surface (affects contact timbre)
        if self.randomize_surface:
            surface = self.rng.choice(self.MATERIAL_CLASSES)
        else:
            surface = self.surface

        # 3. Randomize source position (or use fixed for pure material task)
        if self.randomize_position:
            distance = self.rng.uniform(1.5, 3.0)  # Narrower range for classification
            azimuth = self.rng.uniform(-180.0, 180.0)
        else:
            distance = 2.0  # Fixed distance
            azimuth = 0.0   # Directly in front

        elevation = 0.0  # Horizontal plane

        # Convert spherical → Cartesian
        az_rad = np.deg2rad(azimuth)
        el_rad = np.deg2rad(elevation)
        dx = distance * np.cos(el_rad) * np.cos(az_rad)
        dy = distance * np.cos(el_rad) * np.sin(az_rad)
        dz = distance * np.sin(el_rad)

        self.source_pos = self.mic.position + np.array([dx, dy, dz])

        # 4. Generate audio with selected material
        mono_audio = self._generate_impact_sound(self.true_material, surface)

        # 5. Render to binaural stereo
        stereo = render_spatial_source(mono_audio, self.source_pos, self.mic, self.hrtf)

        # Truncate/pad to desired duration
        target_samples = int(self.audio_duration * self.sr)
        if len(stereo) > target_samples:
            stereo = stereo[:target_samples]
        elif len(stereo) < target_samples:
            stereo = np.pad(stereo, ((0, target_samples - len(stereo)), (0, 0)))

        self.current_audio = stereo.astype(np.float32)

        # 6. Extract observation
        if self.use_mel:
            obs = extract_mel_spectrogram(
                self.current_audio,
                sr=self.sr,
                n_mels=self.n_mels
            )
        else:
            obs = self.current_audio

        info = {
            "true_material": self.true_material,
            "true_material_idx": self.true_material_idx,
            "surface": surface,
            "source_pos": self.source_pos.copy(),
            "distance": distance,
            "azimuth": azimuth
        }

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Agent predicts material class."""
        predicted_material_idx = int(action)
        predicted_material = self.MATERIAL_CLASSES[predicted_material_idx]

        # Compute reward
        reward = classification_reward(predicted_material_idx, self.true_material_idx)

        # Update confusion matrix
        self.confusion_counts[self.true_material_idx, predicted_material_idx] += 1

        terminated = True  # Single-step prediction
        truncated = False

        info = {
            "true_material": self.true_material,
            "true_material_idx": self.true_material_idx,
            "predicted_material": predicted_material,
            "predicted_material_idx": predicted_material_idx,
            "correct": (predicted_material_idx == self.true_material_idx)
        }

        # Return same observation (doesn't change)
        if self.use_mel:
            obs = extract_mel_spectrogram(self.current_audio, self.sr, self.n_mels)
        else:
            obs = self.current_audio

        return obs, reward, terminated, truncated, info

    def _generate_impact_sound(self, material: str, surface: str) -> np.ndarray:
        """Generate impact sound for given material."""
        spec = SceneSpec(
            kind=self.scenario_type,
            surface=surface,
            obj=material,
            duration=self.audio_duration,
            seed=int(self.rng.integers(0, 2**31))
        )

        # Simulate physics
        tele = simulate(spec)
        events = extract_events(tele)

        # Synthesize audio
        mat = MATERIALS[material]
        bank = sample_modes(mat, self.rng, size=1.0)
        out = np.zeros(int(self.audio_duration * self.sr))

        for ev in events:
            render_impact(out, ev["t"], ev["J"], 0.2, bank, mat)

        mono = limit(out)

        return mono

    def get_confusion_matrix(self) -> np.ndarray:
        """Return confusion matrix: [true_class, predicted_class]."""
        return self.confusion_counts.copy()

    def reset_confusion_matrix(self):
        """Reset confusion matrix tracking."""
        self.confusion_counts = np.zeros((4, 4), dtype=int)
