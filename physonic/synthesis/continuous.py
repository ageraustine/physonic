"""Continuous sound synthesis for equipment sounds.

This module synthesizes continuous equipment sounds (fans, servers, HVAC) using
additive synthesis with harmonics, filtered noise, and amplitude modulation.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import scipy.signal


@dataclass
class ContinuousSoundSpec:
    """Specification for continuous equipment sound.

    Attributes:
        source_id: Unique identifier (e.g., "server_rack_1")
        sound_type: Type of equipment sound ("fan", "server_hum", "cooling_system", "hvac")
        fundamental_freq: Base frequency in Hz (e.g., 120Hz for server, 60Hz for fan)
        harmonics: Amplitudes for harmonics [1.0, 0.5, 0.3] means fundamental + 2 harmonics
        noise_amount: Amount of broadband noise (0.0-1.0)
        modulation_freq: Slow amplitude modulation frequency in Hz (for pulsing effect)
        modulation_depth: Amplitude modulation depth (0.0-1.0)
        position: 3D position (x, y, z) in meters
        volume: Relative loudness (0.0-1.0)
    """
    source_id: str
    sound_type: str
    fundamental_freq: float
    harmonics: List[float]
    noise_amount: float
    modulation_freq: float
    modulation_depth: float
    position: Tuple[float, float, float]
    volume: float = 1.0


# Preset configurations for common equipment sounds
SOUND_PRESETS = {
    "server_hum": {
        "fundamental_freq": 120.0,
        "harmonics": [1.0, 0.3, 0.15, 0.08],
        "noise_amount": 0.15,
        "modulation_freq": 0.5,
        "modulation_depth": 0.1,
    },
    "fan": {
        "fundamental_freq": 80.0,
        "harmonics": [1.0, 0.5, 0.3, 0.2, 0.1],
        "noise_amount": 0.35,
        "modulation_freq": 1.5,
        "modulation_depth": 0.25,
    },
    "cooling_system": {
        "fundamental_freq": 180.0,
        "harmonics": [1.0, 0.6, 0.4, 0.25],
        "noise_amount": 0.45,
        "modulation_freq": 2.0,
        "modulation_depth": 0.35,
    },
    "hvac": {
        "fundamental_freq": 60.0,
        "harmonics": [1.0, 0.4, 0.2, 0.12, 0.08],
        "noise_amount": 0.5,
        "modulation_freq": 0.8,
        "modulation_depth": 0.3,
    },
}


def create_preset_sound(
    source_id: str,
    sound_type: str,
    position: Tuple[float, float, float],
    volume: float = 1.0,
) -> ContinuousSoundSpec:
    """Create a continuous sound spec from a preset.

    Args:
        source_id: Unique identifier for this sound source
        sound_type: One of "server_hum", "fan", "cooling_system", "hvac"
        position: 3D position (x, y, z) in meters
        volume: Relative loudness (0.0-1.0)

    Returns:
        ContinuousSoundSpec with preset parameters

    Raises:
        ValueError: If sound_type is not in SOUND_PRESETS
    """
    if sound_type not in SOUND_PRESETS:
        raise ValueError(
            f"Unknown sound type '{sound_type}'. "
            f"Available presets: {list(SOUND_PRESETS.keys())}"
        )

    preset = SOUND_PRESETS[sound_type]

    return ContinuousSoundSpec(
        source_id=source_id,
        sound_type=sound_type,
        fundamental_freq=preset["fundamental_freq"],
        harmonics=preset["harmonics"],
        noise_amount=preset["noise_amount"],
        modulation_freq=preset["modulation_freq"],
        modulation_depth=preset["modulation_depth"],
        position=position,
        volume=volume,
    )


def synthesize_continuous_sound(
    spec: ContinuousSoundSpec,
    duration: float,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Generate continuous equipment sound using additive synthesis.

    Method:
    1. Generate fundamental tone + harmonics using sine waves
    2. Add filtered noise (bandpass around fundamental frequency)
    3. Apply amplitude modulation for "pulsing" effect
    4. Normalize and return mono audio

    Args:
        spec: Sound specification with all parameters
        duration: Duration in seconds
        sample_rate: Sample rate in Hz (default: 44100)

    Returns:
        Mono audio array of shape (num_samples,)
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # 1. Generate harmonic content (additive synthesis)
    audio = np.zeros(num_samples)

    for harmonic_idx, amplitude in enumerate(spec.harmonics):
        if amplitude > 0:
            freq = spec.fundamental_freq * (harmonic_idx + 1)
            # Add slight frequency wobble for realism (±0.5 Hz)
            phase_noise = np.cumsum(np.random.randn(num_samples) * 0.01 / sample_rate)
            audio += amplitude * np.sin(2 * np.pi * freq * t + phase_noise)

    # 2. Add filtered noise
    if spec.noise_amount > 0:
        noise = np.random.randn(num_samples) * spec.noise_amount

        # Bandpass filter around fundamental frequency
        # Filter range: [fundamental/2, fundamental*4]
        nyquist = sample_rate / 2
        low_freq = max(20, spec.fundamental_freq / 2)
        high_freq = min(nyquist * 0.95, spec.fundamental_freq * 4)

        # Butterworth bandpass filter
        sos = scipy.signal.butter(
            4,
            [low_freq, high_freq],
            btype='band',
            fs=sample_rate,
            output='sos'
        )
        filtered_noise = scipy.signal.sosfilt(sos, noise)

        audio += filtered_noise

    # 3. Apply amplitude modulation (pulsing effect)
    if spec.modulation_depth > 0:
        # LFO (low-frequency oscillator) for amplitude modulation
        lfo = 1.0 - spec.modulation_depth + spec.modulation_depth * (
            0.5 + 0.5 * np.sin(2 * np.pi * spec.modulation_freq * t)
        )
        audio *= lfo

    # 4. Apply volume and normalize
    audio *= spec.volume

    # Normalize to prevent clipping (leave some headroom)
    max_amplitude = np.abs(audio).max()
    if max_amplitude > 0:
        audio = audio / max_amplitude * 0.8

    return audio.astype(np.float32)


def synthesize_multiple_sources(
    specs: List[ContinuousSoundSpec],
    duration: float,
    sample_rate: int = 44100,
    mix_level: float = 0.5,
) -> np.ndarray:
    """Synthesize and mix multiple continuous sound sources.

    Each source is synthesized independently and mixed together with
    automatic gain compensation to prevent clipping.

    Args:
        specs: List of sound specifications
        duration: Duration in seconds
        sample_rate: Sample rate in Hz (default: 44100)
        mix_level: Overall mix level (0.0-1.0, default: 0.5 for background sounds)

    Returns:
        Mixed mono audio array of shape (num_samples,)
    """
    if not specs:
        # Return silence
        return np.zeros(int(duration * sample_rate), dtype=np.float32)

    # Synthesize each source
    sources = [
        synthesize_continuous_sound(spec, duration, sample_rate)
        for spec in specs
    ]

    # Mix with automatic gain compensation
    num_sources = len(sources)
    mixed = np.sum(sources, axis=0) / np.sqrt(num_sources)  # RMS-based mixing

    # Apply overall mix level
    mixed *= mix_level

    # Final normalization with headroom
    max_amplitude = np.abs(mixed).max()
    if max_amplitude > 0:
        mixed = mixed / max_amplitude * 0.8

    return mixed.astype(np.float32)


# Example usage
if __name__ == "__main__":
    import soundfile as sf

    print("Continuous Sound Synthesis Examples")
    print("=" * 60)

    # Example 1: Single server hum
    server_spec = create_preset_sound(
        source_id="server_rack_1",
        sound_type="server_hum",
        position=(0, 0, 0),
        volume=1.0
    )

    print(f"\nGenerating {server_spec.sound_type}...")
    server_audio = synthesize_continuous_sound(server_spec, duration=3.0)
    sf.write("outputs/test_server_hum.wav", server_audio, 44100)
    print(f"  Saved to outputs/test_server_hum.wav ({len(server_audio)} samples)")

    # Example 2: Multiple sources (data center ambient)
    print("\nGenerating data center ambient (3 servers + 2 fans + 1 cooling)...")

    sources = [
        create_preset_sound(f"server_{i}", "server_hum", (i*2, 0, 0), volume=0.7)
        for i in range(3)
    ] + [
        create_preset_sound(f"fan_{i}", "fan", (i*3, 5, 2), volume=0.6)
        for i in range(2)
    ] + [
        create_preset_sound("cooling_1", "cooling_system", (10, 10, 0), volume=0.8)
    ]

    mixed_audio = synthesize_multiple_sources(sources, duration=3.0, mix_level=0.5)
    sf.write("outputs/test_datacenter_ambient.wav", mixed_audio, 44100)
    print(f"  Saved to outputs/test_datacenter_ambient.wav ({len(mixed_audio)} samples)")

    # Example 3: All preset types
    print("\nGenerating all preset types...")
    for sound_type in SOUND_PRESETS.keys():
        spec = create_preset_sound(f"test_{sound_type}", sound_type, (0, 0, 0))
        audio = synthesize_continuous_sound(spec, duration=2.0)
        filename = f"outputs/test_{sound_type}.wav"
        sf.write(filename, audio, 44100)
        print(f"  {sound_type}: {filename}")

    print("\nDone! Check outputs/ directory for generated audio files.")
