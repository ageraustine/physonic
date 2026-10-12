"""Sound selective response RL environment."""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional

from physonic.spatial import load_kemar_hrtf, BinauralMicrophone, render_spatial_source
from physonic.contact.rig import SceneSpec, simulate, extract_events
from physonic.contact.modal_synth import sample_modes, render_impact, limit
from physonic.contact.materials import MATERIALS
from physonic.envs.preprocessing import extract_mel_spectrogram


class SelectiveResponseEnv(gym.Env):
    """
    Selective response to task-relevant sounds.

    Task: Agent hears a sound and must decide whether to respond or ignore it.

    Observation:
        - Mel-spectrogram (n_mels, time_frames, 2) for each ear
        - OR: Raw binaural audio (T, 2) stereo @ 44.1kHz

    Action:
        - Discrete(2): {0: IGNORE, 1: RESPOND}

    Reward:
        - True Positive (correct RESPOND): +1.0
        - True Negative (correct IGNORE): +0.5 (or configurable)
        - False Positive (incorrect RESPOND to distractor): -1.0
        - False Negative (incorrect IGNORE of target): -2.0 (worse penalty)

    Episode:
        - Single-step decision task
        - Each episode: random sound (target or distractor) → agent decides → evaluate
    """

    metadata = {"render_modes": ["none"]}

    # All material classes
    ALL_MATERIALS = ["steel", "glass", "wood", "rubber"]

    def __init__(
        self,
        target_materials: list = ["glass"],  # Materials that require response
        audio_duration: float = 1.0,
        sample_rate: int = 44100,
        use_mel_spectrogram: bool = True,
        n_mels: int = 64,
        scenario_type: str = "drop",
        surface: str = "glass",
        randomize_surface: bool = False,
        randomize_position: bool = True,
        target_probability: float = 0.3,  # 30% targets, 70% distractors
        reward_tp: float = 1.0,          # True positive reward
        reward_tn: float = 0.5,          # True negative reward
        reward_fp: float = -1.0,         # False positive penalty
        reward_fn: float = -2.0,         # False negative penalty (worse)
        hrtf_path: str = "assets/hrtf/mit_kemar/",
        seed: Optional[int] = None
    ):
        super().__init__()

        # Validate target materials
        for mat in target_materials:
            if mat not in self.ALL_MATERIALS:
                raise ValueError(f"Invalid target material: {mat}")

        self.target_materials = target_materials
        self.distractor_materials = [m for m in self.ALL_MATERIALS if m not in target_materials]

        if not self.distractor_materials:
            raise ValueError("Must have at least one distractor material")

        self.audio_duration = audio_duration
        self.sr = sample_rate
        self.use_mel = use_mel_spectrogram
        self.n_mels = n_mels
        self.scenario_type = scenario_type
        self.surface = surface
        self.randomize_surface = randomize_surface
        self.randomize_position = randomize_position
        self.target_probability = target_probability

        # Reward structure
        self.reward_tp = reward_tp
        self.reward_tn = reward_tn
        self.reward_fp = reward_fp
        self.reward_fn = reward_fn

        # Load HRTF
        self.hrtf = load_kemar_hrtf(hrtf_path)

        # Microphone
        self.mic = BinauralMicrophone(
            position=np.array([0.0, 0.0, 1.5]),
            orientation=0.0
        )

        # RNG
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # Observation space
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

        # Action space: binary decision
        self.action_space = gym.spaces.Discrete(2)  # 0=IGNORE, 1=RESPOND

        # Episode state
        self.is_target = None
        self.current_material = None
        self.current_audio = None
        self.source_pos = None

        # Tracking metrics
        self.reset_metrics()

    def reset_metrics(self):
        """Reset precision/recall tracking."""
        self.tp_count = 0  # True positives
        self.tn_count = 0  # True negatives
        self.fp_count = 0  # False positives
        self.fn_count = 0  # False negatives

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate new audio (target or distractor)."""
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 1. Decide if this is a target or distractor
        self.is_target = self.rng.random() < self.target_probability

        # 2. Select material based on target/distractor
        if self.is_target:
            self.current_material = self.rng.choice(self.target_materials)
        else:
            self.current_material = self.rng.choice(self.distractor_materials)

        # 3. Optionally randomize surface
        if self.randomize_surface:
            surface = self.rng.choice(self.ALL_MATERIALS)
        else:
            surface = self.surface

        # 4. Randomize source position
        if self.randomize_position:
            distance = self.rng.uniform(1.5, 3.0)
            azimuth = self.rng.uniform(-180.0, 180.0)
        else:
            distance = 2.0
            azimuth = 0.0

        elevation = 0.0

        # Convert spherical → Cartesian
        az_rad = np.deg2rad(azimuth)
        el_rad = np.deg2rad(elevation)
        dx = distance * np.cos(el_rad) * np.cos(az_rad)
        dy = distance * np.cos(el_rad) * np.sin(az_rad)
        dz = distance * np.sin(el_rad)

        self.source_pos = self.mic.position + np.array([dx, dy, dz])

        # 5. Generate audio
        mono_audio = self._generate_impact_sound(self.current_material, surface)

        # 6. Render to binaural stereo
        stereo = render_spatial_source(mono_audio, self.source_pos, self.mic, self.hrtf)

        # Truncate/pad
        target_samples = int(self.audio_duration * self.sr)
        if len(stereo) > target_samples:
            stereo = stereo[:target_samples]
        elif len(stereo) < target_samples:
            stereo = np.pad(stereo, ((0, target_samples - len(stereo)), (0, 0)))

        self.current_audio = stereo.astype(np.float32)

        # 7. Extract observation
        if self.use_mel:
            obs = extract_mel_spectrogram(
                self.current_audio,
                sr=self.sr,
                n_mels=self.n_mels
            )
        else:
            obs = self.current_audio

        info = {
            "is_target": self.is_target,
            "material": self.current_material,
            "surface": surface,
            "source_pos": self.source_pos.copy(),
            "distance": distance,
            "azimuth": azimuth
        }

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Agent decides: IGNORE (0) or RESPOND (1)."""
        agent_responds = bool(action)

        # Determine outcome
        if self.is_target and agent_responds:
            # True Positive: correctly responded to target
            reward = self.reward_tp
            outcome = "TP"
            self.tp_count += 1
        elif not self.is_target and not agent_responds:
            # True Negative: correctly ignored distractor
            reward = self.reward_tn
            outcome = "TN"
            self.tn_count += 1
        elif not self.is_target and agent_responds:
            # False Positive: incorrectly responded to distractor
            reward = self.reward_fp
            outcome = "FP"
            self.fp_count += 1
        else:  # self.is_target and not agent_responds
            # False Negative: incorrectly ignored target
            reward = self.reward_fn
            outcome = "FN"
            self.fn_count += 1

        terminated = True
        truncated = False

        info = {
            "is_target": self.is_target,
            "agent_responds": agent_responds,
            "material": self.current_material,
            "outcome": outcome,
            "reward": reward
        }

        # Return same observation
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

        tele = simulate(spec)
        events = extract_events(tele)

        mat = MATERIALS[material]
        bank = sample_modes(mat, self.rng, size=1.0)
        out = np.zeros(int(self.audio_duration * self.sr))

        for ev in events:
            render_impact(out, ev["t"], ev["J"], 0.2, bank, mat)

        mono = limit(out)
        return mono

    def get_metrics(self) -> Dict[str, float]:
        """Compute precision, recall, F1 score."""
        precision = self.tp_count / (self.tp_count + self.fp_count) if (self.tp_count + self.fp_count) > 0 else 0.0
        recall = self.tp_count / (self.tp_count + self.fn_count) if (self.tp_count + self.fn_count) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        accuracy = (self.tp_count + self.tn_count) / (self.tp_count + self.tn_count + self.fp_count + self.fn_count) if (self.tp_count + self.tn_count + self.fp_count + self.fn_count) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": self.tp_count,
            "tn": self.tn_count,
            "fp": self.fp_count,
            "fn": self.fn_count
        }

    def get_confusion_matrix(self) -> np.ndarray:
        """Return 2x2 confusion matrix: [[TN, FP], [FN, TP]]."""
        return np.array([
            [self.tn_count, self.fp_count],
            [self.fn_count, self.tp_count]
        ])
