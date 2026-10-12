"""Sound mixing utilities for combining impact, continuous, and alert sounds.

This module provides functions to mix different types of acoustic sources:
- Impact sounds (from modal synthesis)
- Continuous equipment sounds (background)
- Alert/beep sounds (foreground events)
"""

from typing import List, Tuple
import numpy as np

from physonic.synthesis.continuous import ContinuousSoundSpec, synthesize_continuous_sound
from physonic.synthesis.alerts import AlertSoundSpec, synthesize_alert_sound, place_alert_in_timeline


def mix_acoustic_scene(
    impact_sounds: List[Tuple[np.ndarray, float]],
    continuous_sounds: List[ContinuousSoundSpec],
    alert_sounds: List[Tuple[AlertSoundSpec, float]],
    duration: float,
    sample_rate: int = 44100,
    continuous_level: float = 0.3,
    alert_level: float = 0.6,
    impact_level: float = 1.0,
) -> np.ndarray:
    """Mix all sound sources into final mono audio.

    Mixing strategy:
    - Continuous sounds: Background level (default -10dB = 0.3)
    - Alert sounds: Mid-level, noticeable (default -4dB = 0.6)
    - Impact sounds: Foreground events (default 0dB = 1.0)

    Steps:
    1. Generate continuous sounds for full duration (background loop)
    2. Synthesize alert sounds at specified trigger times
    3. Place impact sounds at specified times
    4. Sum all sources with appropriate mixing levels
    5. Apply normalization/limiting

    Args:
        impact_sounds: List of (audio_array, start_time) from modal synthesis
        continuous_sounds: List of continuous sound specs for background
        alert_sounds: List of (alert_spec, trigger_time) for alert events
        duration: Total audio duration in seconds
        sample_rate: Sample rate in Hz (default: 44100)
        continuous_level: Mix level for continuous sounds (0.0-1.0, default: 0.3)
        alert_level: Mix level for alert sounds (0.0-1.0, default: 0.6)
        impact_level: Mix level for impact sounds (0.0-1.0, default: 1.0)

    Returns:
        Mixed mono audio array of shape (num_samples,)
    """
    num_samples = int(duration * sample_rate)
    mixed = np.zeros(num_samples, dtype=np.float32)

    # 1. Add continuous sounds (background)
    if continuous_sounds:
        for spec in continuous_sounds:
            continuous_audio = synthesize_continuous_sound(spec, duration, sample_rate)
            continuous_audio *= continuous_level
            mixed += continuous_audio

    # 2. Add alert sounds at specified times
    if alert_sounds:
        for alert_spec, trigger_time in alert_sounds:
            alert_audio = synthesize_alert_sound(alert_spec, sample_rate)
            alert_audio *= alert_level

            # Place in timeline
            timeline = place_alert_in_timeline(alert_audio, trigger_time, duration, sample_rate)
            mixed += timeline

    # 3. Add impact sounds at specified times
    if impact_sounds:
        for impact_audio, start_time in impact_sounds:
            impact_audio = np.array(impact_audio, dtype=np.float32)
            impact_audio *= impact_level

            start_sample = int(start_time * sample_rate)
            impact_len = len(impact_audio)
            end_sample = min(start_sample + impact_len, num_samples)
            actual_len = end_sample - start_sample

            if actual_len > 0:
                mixed[start_sample:end_sample] += impact_audio[:actual_len]

    # 4. Normalize to prevent clipping (with headroom)
    max_amplitude = np.abs(mixed).max()
    if max_amplitude > 0:
        # Soft limiting: compress peaks above 0.8
        threshold = 0.8
        if max_amplitude > threshold:
            # Apply soft clipping to peaks
            over_threshold = np.abs(mixed) > threshold
            mixed[over_threshold] = np.sign(mixed[over_threshold]) * (
                threshold + (np.abs(mixed[over_threshold]) - threshold) / 3
            )

        # Final normalization with headroom
        max_amplitude = np.abs(mixed).max()
        mixed = mixed / max_amplitude * 0.9

    return mixed.astype(np.float32)


def mix_with_layers(
    layers: List[Tuple[np.ndarray, float]],
    duration: float,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Mix multiple audio layers with individual gain control.

    This is a simpler mixing function for pre-rendered audio layers.

    Args:
        layers: List of (audio_array, gain) pairs
        duration: Total duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Mixed mono audio array
    """
    num_samples = int(duration * sample_rate)
    mixed = np.zeros(num_samples, dtype=np.float32)

    for audio, gain in layers:
        audio = np.array(audio, dtype=np.float32) * gain

        # Add to mix (truncate or pad as needed)
        audio_len = len(audio)
        if audio_len <= num_samples:
            mixed[:audio_len] += audio
        else:
            mixed += audio[:num_samples]

    # Normalize
    max_amplitude = np.abs(mixed).max()
    if max_amplitude > 0:
        mixed = mixed / max_amplitude * 0.9

    return mixed


def create_data_center_ambient(
    num_servers: int = 8,
    num_fans: int = 3,
    num_cooling: int = 2,
    duration: float = 5.0,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Create realistic data center ambient sound.

    This is a convenience function for generating typical data center background.

    Args:
        num_servers: Number of server racks
        num_fans: Number of fans
        num_cooling: Number of cooling systems
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Mixed ambient audio
    """
    from physonic.synthesis.continuous import create_preset_sound

    continuous_sounds = []

    # Add servers
    for i in range(num_servers):
        spec = create_preset_sound(
            source_id=f"server_{i}",
            sound_type="server_hum",
            position=(i * 2, 0, 0),
            volume=0.6 + np.random.rand() * 0.2  # Slight variation
        )
        continuous_sounds.append(spec)

    # Add fans
    for i in range(num_fans):
        spec = create_preset_sound(
            source_id=f"fan_{i}",
            sound_type="fan",
            position=(i * 3, 5, 2),
            volume=0.5 + np.random.rand() * 0.3
        )
        continuous_sounds.append(spec)

    # Add cooling systems
    for i in range(num_cooling):
        spec = create_preset_sound(
            source_id=f"cooling_{i}",
            sound_type="cooling_system",
            position=(i * 8, 8, 0),
            volume=0.7 + np.random.rand() * 0.2
        )
        continuous_sounds.append(spec)

    # Mix with no impacts or alerts (pure ambient)
    return mix_acoustic_scene(
        impact_sounds=[],
        continuous_sounds=continuous_sounds,
        alert_sounds=[],
        duration=duration,
        sample_rate=sample_rate,
        continuous_level=0.4,
    )


# Example usage
if __name__ == "__main__":
    import soundfile as sf
    from physonic.synthesis.continuous import create_preset_sound
    from physonic.synthesis.alerts import create_preset_alert

    print("Sound Mixing Examples")
    print("=" * 60)

    # Example 1: Data center ambient with alerts
    print("\nExample 1: Data center with maintenance alerts...")

    # Background: 3 servers + 2 fans
    continuous_sounds = [
        create_preset_sound(f"server_{i}", "server_hum", (i*2, 0, 0), volume=0.6)
        for i in range(3)
    ] + [
        create_preset_sound(f"fan_{i}", "fan", (i*3, 5, 2), volume=0.5)
        for i in range(2)
    ]

    # Events: Beep at 1s, warning at 3s
    alert_sounds = [
        (create_preset_alert("beep", (5, 0, 0)), 1.0),
        (create_preset_alert("warning", (10, 0, 0)), 3.0),
    ]

    # Simulate impact sound (just noise for demo)
    impact_audio = np.random.randn(4410) * 0.3  # 100ms of noise
    impact_sounds = [(impact_audio, 2.0)]  # Impact at 2 seconds

    mixed = mix_acoustic_scene(
        impact_sounds=impact_sounds,
        continuous_sounds=continuous_sounds,
        alert_sounds=alert_sounds,
        duration=5.0,
    )

    sf.write("outputs/test_mixed_datacenter.wav", mixed, 44100)
    print(f"  Saved to outputs/test_mixed_datacenter.wav ({len(mixed)} samples)")
    print(f"  Contains: 3 servers, 2 fans, 1 impact (2s), beep (1s), warning (3s)")

    # Example 2: Data center ambient (no events)
    print("\nExample 2: Pure data center ambient (8 servers, 3 fans, 2 cooling)...")
    ambient = create_data_center_ambient(
        num_servers=8,
        num_fans=3,
        num_cooling=2,
        duration=5.0
    )
    sf.write("outputs/test_datacenter_ambient_full.wav", ambient, 44100)
    print(f"  Saved to outputs/test_datacenter_ambient_full.wav ({len(ambient)} samples)")

    # Example 3: Layer mixing
    print("\nExample 3: Simple layer mixing...")
    layer1 = synthesize_continuous_sound(
        create_preset_sound("bg1", "server_hum", (0, 0, 0)),
        duration=3.0
    )
    layer2 = synthesize_continuous_sound(
        create_preset_sound("bg2", "fan", (5, 0, 0)),
        duration=3.0
    )

    layered = mix_with_layers(
        layers=[(layer1, 0.5), (layer2, 0.7)],
        duration=3.0
    )
    sf.write("outputs/test_mixed_layers.wav", layered, 44100)
    print(f"  Saved to outputs/test_mixed_layers.wav ({len(layered)} samples)")

    print("\nDone! Check outputs/ directory for generated audio files.")
