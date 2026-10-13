"""Efficient audio observation generation for RL environments.

This module provides high-performance audio observation generation using:
- Spatial audio culling (only render audible sources)
- Binaural HRTF rendering
- Mimi neural codec compression
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from physonic.synthesis.continuous import (
    ContinuousSoundSpec,
    synthesize_continuous_sound,
)
from physonic.spatial.hrtf import HRTFSet, load_kemar_hrtf
from physonic.spatial.microphone import BinauralMicrophone
from physonic.spatial.renderer import render_spatial_source
from physonic.codecs.engines.mimi import MimiAudioEngine


class AudioObservationGenerator:
    """Generates compressed audio observations for RL environments.

    Features:
    - Lazy-loaded Mimi codec (only loads when first used)
    - Spatial audio culling (only renders nearby sounds)
    - HRTF-based binaural rendering
    - Efficient caching of synthesized sounds
    """

    def __init__(
        self,
        continuous_sounds: List[ContinuousSoundSpec],
        hrtf_path: str,
        audio_duration: float = 0.5,
        sample_rate: int = 44100,
        max_audio_distance: float = 30.0,
        use_mimi: bool = True,
        mimi_parallel: bool = False,
    ):
        """Initialize audio observation generator.

        Args:
            continuous_sounds: List of continuous equipment sound specs
            hrtf_path: Path to HRTF dataset (MIT KEMAR)
            audio_duration: Duration of audio snippets in seconds
            sample_rate: Audio sample rate in Hz
            max_audio_distance: Maximum distance for audio rendering (meters)
            use_mimi: Enable Mimi codec compression
            mimi_parallel: Use parallel Mimi processing (slower on short clips)
        """
        self.continuous_sounds = continuous_sounds
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.max_audio_distance = max_audio_distance
        self.use_mimi = use_mimi

        # Load HRTF (cached)
        self.hrtf = load_kemar_hrtf(hrtf_path)

        # Lazy-load Mimi codec (only when needed)
        self._mimi = None
        self._mimi_parallel = mimi_parallel

        # Cache synthesized mono sounds for each source
        # Key: source_id, Value: mono audio array
        self._sound_cache = {}

    @property
    def mimi(self) -> Optional[MimiAudioEngine]:
        """Lazy-load Mimi codec."""
        if self.use_mimi and self._mimi is None:
            self._mimi = MimiAudioEngine(
                parallel=self._mimi_parallel,
                encode_chunk_length=0,  # No chunking for short clips
                decode_chunk_length=0,
            )
        return self._mimi

    def generate_observation(
        self,
        robot_position: np.ndarray,
        robot_orientation: float,
    ) -> np.ndarray:
        """Generate audio observation from robot's perspective.

        Args:
            robot_position: Robot position [x, y, z] in meters
            robot_orientation: Robot heading in radians

        Returns:
            If use_mimi=True: Mimi codes array of shape (8, T) where T depends on duration
            If use_mimi=False: Binaural audio array of shape (samples, 2)
        """
        # 1. Spatial audio culling - find audible sounds
        audible_sounds = self._cull_sounds(robot_position)

        if len(audible_sounds) == 0:
            # No audible sounds - return silence
            return self._generate_silence()

        # 2. Create binaural microphone at robot position
        # Convert orientation from radians to degrees for microphone
        mic = BinauralMicrophone(
            position=robot_position,
            orientation=np.degrees(robot_orientation),
        )

        # 3. Synthesize and spatially render each sound
        binaural_audio = self._render_binaural(audible_sounds, mic)

        # 4. Encode with Mimi if enabled
        if self.use_mimi:
            encoded = self.mimi.encode(binaural_audio.T, self.sample_rate)
            codes = encoded["codes"][0]  # Shape: (8, time_frames)
            return codes
        else:
            return binaural_audio

    def _cull_sounds(
        self,
        robot_position: np.ndarray,
    ) -> List[Tuple[ContinuousSoundSpec, float]]:
        """Spatial audio culling - filter sounds by distance.

        Args:
            robot_position: Robot position [x, y, z]

        Returns:
            List of (sound_spec, distance) for audible sources, sorted by distance
        """
        audible = []

        for sound_spec in self.continuous_sounds:
            source_position = np.array(sound_spec.position)
            distance = np.linalg.norm(source_position - robot_position)

            if distance <= self.max_audio_distance:
                audible.append((sound_spec, distance))

        # Sort by distance (closest first for better mixing)
        audible.sort(key=lambda x: x[1])

        return audible

    def _render_binaural(
        self,
        audible_sounds: List[Tuple[ContinuousSoundSpec, float]],
        microphone: BinauralMicrophone,
    ) -> np.ndarray:
        """Render binaural audio from audible sources.

        Args:
            audible_sounds: List of (sound_spec, distance) tuples
            microphone: Binaural microphone at robot position

        Returns:
            Binaural audio array of shape (samples, 2)
        """
        # Initialize output buffer
        # Account for HRTF convolution length
        hrtf_length = self.hrtf.hrir_length
        audio_samples = int(self.audio_duration * self.sample_rate)
        output_length = audio_samples + hrtf_length - 1
        binaural_mix = np.zeros((output_length, 2), dtype=np.float32)

        # Render each source
        for sound_spec, distance in audible_sounds:
            # Get or synthesize mono audio
            mono = self._get_cached_sound(sound_spec)

            # Spatially render
            source_position = np.array(sound_spec.position)
            binaural = render_spatial_source(
                mono,
                source_position,
                microphone,
                self.hrtf,
            )

            # Mix into output (trim to output length if needed)
            mix_length = min(binaural.shape[0], output_length)
            binaural_mix[:mix_length] += binaural[:mix_length]

        # Normalize to prevent clipping
        max_val = np.abs(binaural_mix).max()
        if max_val > 0.9:
            binaural_mix = binaural_mix * 0.9 / max_val

        # Trim back to original audio length (remove HRTF tail for consistency)
        binaural_mix = binaural_mix[:audio_samples]

        return binaural_mix

    def _get_cached_sound(self, sound_spec: ContinuousSoundSpec) -> np.ndarray:
        """Get or synthesize and cache mono audio for a sound source.

        Args:
            sound_spec: Sound specification

        Returns:
            Mono audio array of shape (samples,)
        """
        if sound_spec.source_id not in self._sound_cache:
            # Synthesize and cache
            mono = synthesize_continuous_sound(
                sound_spec,
                duration=self.audio_duration,
                sample_rate=self.sample_rate,
            )
            self._sound_cache[sound_spec.source_id] = mono

        return self._sound_cache[sound_spec.source_id]

    def _generate_silence(self) -> np.ndarray:
        """Generate silent observation.

        Returns:
            If use_mimi=True: Mimi codes for silence
            If use_mimi=False: Zeros array
        """
        if self.use_mimi:
            # Encode silence
            audio_samples = int(self.audio_duration * self.sample_rate)
            silence = np.zeros((2, audio_samples), dtype=np.float32)
            encoded = self.mimi.encode(silence, self.sample_rate)
            return encoded["codes"][0]
        else:
            audio_samples = int(self.audio_duration * self.sample_rate)
            return np.zeros((audio_samples, 2), dtype=np.float32)

    def get_observation_shape(self) -> Tuple[int, ...]:
        """Get expected observation shape.

        Returns:
            Shape tuple: (8, time_frames) for Mimi or (samples, 2) for raw audio
        """
        if self.use_mimi:
            # Estimate Mimi frames (12.5 Hz frame rate)
            time_frames = int(self.audio_duration * 12.5)
            return (8, time_frames)
        else:
            audio_samples = int(self.audio_duration * self.sample_rate)
            return (audio_samples, 2)
