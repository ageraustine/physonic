"""Sound source localization RL environment."""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional

from physonic.spatial import load_kemar_hrtf, BinauralMicrophone, render_spatial_source
from physonic.contact.rig import SceneSpec, simulate, extract_events
from physonic.contact.modal_synth import sample_modes, render_impact, limit
from physonic.contact.materials import MATERIALS
from physonic.envs.preprocessing import extract_mel_spectrogram
from physonic.envs.rewards import angular_error


class SoundLocalizationEnv(gym.Env):
    """
    Sound source localization from binaural audio.

    Task: Agent hears a sound from an unknown direction and must predict azimuth angle.

    Observation:
        - Raw binaural audio (T×44100, 2) stereo @ 44.1kHz
        - OR: Mel-spectrogram (n_mels, time_frames, 2) for each ear

    Action:
        - Continuous azimuth angle [-180, 180] degrees
        - 0° = front (+X), 90° = left (+Y), -90° = right (-Y), 180° = back

    Reward:
        - Negative angular error: -|pred - true| / 180
        - Range: [-1, 0] where 0 is perfect prediction

    Episode:
        - Single-step prediction task
        - Each episode: new random source position → audio → agent predicts → evaluate
    """

    metadata = {"render_modes": ["none"]}

    def __init__(
        self,
        audio_duration: float = 1.0,
        sample_rate: int = 44100,
        use_mel_spectrogram: bool = True,
        n_mels: int = 64,
        scenario_type: str = "drop",  # "drop", "slide"
        material: str = "steel",
        surface: str = "glass",
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
        self.material = material
        self.surface = surface
        self.room_acoustics = room_acoustics

        # Load HRTF once (expensive, do in __init__)
        self.hrtf = load_kemar_hrtf(hrtf_path)

        # Microphone at origin, facing forward
        self.mic = BinauralMicrophone(
            position=np.array([0.0, 0.0, 1.5]),  # 1.5m height (human ear level)
            orientation=0.0  # Facing +X
        )

        # RNG for reproducibility
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # Observation space
        if use_mel_spectrogram:
            # Mel-spectrogram: (n_mels, time_frames, 2)
            # Approximate time_frames for 1s audio @ hop_length=512
            time_frames = int(np.ceil(self.audio_duration * self.sr / 512))
            self.observation_space = gym.spaces.Box(
                low=-80.0,  # dB scale
                high=40.0,
                shape=(self.n_mels, time_frames, 2),
                dtype=np.float32
            )
        else:
            # Raw audio: (T, 2) stereo
            audio_samples = int(self.audio_duration * self.sr)
            self.observation_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(audio_samples, 2),
                dtype=np.float32
            )

        # Action space: continuous azimuth angle [-180, 180]
        self.action_space = gym.spaces.Box(
            low=-180.0,
            high=180.0,
            shape=(1,),
            dtype=np.float32
        )

        # Episode state
        self.true_azimuth = None
        self.true_elevation = None
        self.source_pos = None
        self.current_audio = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate new audio from random source position."""
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 1. Sample random source position (azimuth-only, variable distance)
        distance = self.rng.uniform(1.0, 5.0)  # 1-5m from mic
        azimuth = self.rng.uniform(-180.0, 180.0)
        elevation = 0.0  # Horizontal plane for Phase 4A

        # Convert spherical → Cartesian (relative to mic)
        # Mic at (0, 0, 1.5), source at spherical offset
        az_rad = np.deg2rad(azimuth)
        el_rad = np.deg2rad(elevation)

        dx = distance * np.cos(el_rad) * np.cos(az_rad)
        dy = distance * np.cos(el_rad) * np.sin(az_rad)
        dz = distance * np.sin(el_rad)

        self.source_pos = self.mic.position + np.array([dx, dy, dz])
        self.true_azimuth = azimuth
        self.true_elevation = elevation

        # 2. Generate mono audio (impact sound)
        mono_audio = self._generate_impact_sound()

        # 3. Render to binaural stereo
        stereo = render_spatial_source(mono_audio, self.source_pos, self.mic, self.hrtf)

        # Truncate/pad to desired duration
        target_samples = int(self.audio_duration * self.sr)
        if len(stereo) > target_samples:
            stereo = stereo[:target_samples]
        elif len(stereo) < target_samples:
            stereo = np.pad(stereo, ((0, target_samples - len(stereo)), (0, 0)))

        self.current_audio = stereo.astype(np.float32)

        # 4. Extract observation (raw or mel-spectrogram)
        if self.use_mel:
            obs = extract_mel_spectrogram(
                self.current_audio,
                sr=self.sr,
                n_mels=self.n_mels
            )
        else:
            obs = self.current_audio

        info = {
            "true_azimuth": self.true_azimuth,
            "true_elevation": self.true_elevation,
            "source_pos": self.source_pos.copy(),  # Copy to avoid sharing references
            "distance": distance
        }

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Agent predicts azimuth angle."""
        predicted_azimuth = float(action[0])

        # Compute angular error
        error = angular_error(predicted_azimuth, self.true_azimuth)

        # Reward: negative normalized error [-1, 0]
        reward = -error / 180.0

        terminated = True  # Single-step prediction
        truncated = False

        info = {
            "true_azimuth": self.true_azimuth,
            "predicted_azimuth": predicted_azimuth,
            "angular_error": error,
            "source_pos": self.source_pos.copy()  # Copy to avoid sharing references
        }

        # Return same observation (doesn't change)
        if self.use_mel:
            obs = extract_mel_spectrogram(self.current_audio, self.sr, self.n_mels)
        else:
            obs = self.current_audio

        return obs, reward, terminated, truncated, info

    def _generate_impact_sound(self) -> np.ndarray:
        """Generate synthetic impact sound using modal synthesis."""
        # Create simple drop scenario
        spec = SceneSpec(
            kind=self.scenario_type,
            surface=self.surface,
            obj=self.material,
            duration=self.audio_duration,
            seed=int(self.rng.integers(0, 2**31))
        )

        # Simulate physics
        tele = simulate(spec)
        events = extract_events(tele)

        # Synthesize audio
        mat = MATERIALS[self.material]
        bank = sample_modes(mat, self.rng, size=1.0)
        out = np.zeros(int(self.audio_duration * self.sr))

        for ev in events:
            render_impact(out, ev["t"], ev["J"], 0.2, bank, mat)

        mono = limit(out)

        # Optional: apply room acoustics (Phase 4 later)
        if self.room_acoustics:
            # Room acoustics currently applies after spatial rendering
            # Would need refactoring to apply before
            pass

        return mono
