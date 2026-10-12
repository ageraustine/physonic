"""Alert and beep sound synthesis.

This module synthesizes alert/beep sounds (alarms, warnings, status beeps) using
pure tone synthesis with ADSR envelopes.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class AlertSoundSpec:
    """Specification for alert/beep sound.

    Attributes:
        alert_type: Type of alert ("beep", "alarm", "warning", "chime", "error")
        frequency: Tone frequency in Hz (e.g., 1000Hz for beep)
        duration: Single beep duration in seconds
        pattern: Beep pattern ("single", "double", "triple", "continuous")
        envelope: Envelope type ("sharp" for fast attack/release, "soft" for gradual)
        position: 3D position (x, y, z) in meters
        volume: Relative loudness (0.0-1.0)
        gap_duration: Time between beeps in pattern (seconds, default: 0.2)
    """
    alert_type: str
    frequency: float
    duration: float
    pattern: str
    envelope: str
    position: Tuple[float, float, float]
    volume: float = 1.0
    gap_duration: float = 0.2


# Preset configurations for common alert sounds
ALERT_PRESETS = {
    "beep": {
        "frequency": 1000.0,
        "duration": 0.1,
        "pattern": "single",
        "envelope": "sharp",
    },
    "double_beep": {
        "frequency": 1000.0,
        "duration": 0.1,
        "pattern": "double",
        "envelope": "sharp",
    },
    "alarm": {
        "frequency": 800.0,
        "duration": 0.5,
        "pattern": "continuous",
        "envelope": "sharp",
    },
    "warning": {
        "frequency": 600.0,
        "duration": 0.3,
        "pattern": "triple",
        "envelope": "soft",
    },
    "chime": {
        "frequency": 523.25,  # C5
        "duration": 0.3,
        "pattern": "single",
        "envelope": "soft",
    },
    "error": {
        "frequency": 500.0,
        "duration": 0.2,
        "pattern": "triple",
        "envelope": "sharp",
    },
}


def create_preset_alert(
    alert_type: str,
    position: Tuple[float, float, float],
    volume: float = 1.0,
) -> AlertSoundSpec:
    """Create an alert sound spec from a preset.

    Args:
        alert_type: One of "beep", "double_beep", "alarm", "warning", "chime", "error"
        position: 3D position (x, y, z) in meters
        volume: Relative loudness (0.0-1.0)

    Returns:
        AlertSoundSpec with preset parameters

    Raises:
        ValueError: If alert_type is not in ALERT_PRESETS
    """
    if alert_type not in ALERT_PRESETS:
        raise ValueError(
            f"Unknown alert type '{alert_type}'. "
            f"Available presets: {list(ALERT_PRESETS.keys())}"
        )

    preset = ALERT_PRESETS[alert_type]

    return AlertSoundSpec(
        alert_type=alert_type,
        frequency=preset["frequency"],
        duration=preset["duration"],
        pattern=preset["pattern"],
        envelope=preset["envelope"],
        position=position,
        volume=volume,
    )


def _apply_adsr_envelope(
    audio: np.ndarray,
    sample_rate: int,
    envelope_type: str,
) -> np.ndarray:
    """Apply ADSR envelope to audio signal.

    Args:
        audio: Input audio array
        sample_rate: Sample rate in Hz
        envelope_type: "sharp" (fast attack/release) or "soft" (gradual)

    Returns:
        Audio with envelope applied
    """
    num_samples = len(audio)

    if envelope_type == "sharp":
        # Fast attack/release: 5ms attack, no decay/sustain, 10ms release
        attack_time = 0.005
        release_time = 0.01
    else:  # "soft"
        # Gradual attack/release: 50ms attack, 30ms release
        attack_time = 0.05
        release_time = 0.03

    attack_samples = int(attack_time * sample_rate)
    release_samples = int(release_time * sample_rate)

    # Ensure attack/release don't exceed signal length
    attack_samples = min(attack_samples, num_samples // 3)
    release_samples = min(release_samples, num_samples // 3)

    # Create envelope
    envelope = np.ones(num_samples)

    # Attack (linear ramp up)
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

    # Release (linear ramp down)
    if release_samples > 0:
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)

    return audio * envelope


def synthesize_alert_sound(
    spec: AlertSoundSpec,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Generate alert/beep sound using sine wave + ADSR envelope.

    Method:
    1. Generate pure sine wave at specified frequency
    2. Apply ADSR envelope (attack, decay, sustain, release)
    3. Repeat for pattern (single/double/triple beeps)
    4. Return mono audio

    Args:
        spec: Alert sound specification
        sample_rate: Sample rate in Hz (default: 44100)

    Returns:
        Mono audio array of shape (num_samples,)
    """
    # Determine number of beeps
    if spec.pattern == "single":
        num_beeps = 1
    elif spec.pattern == "double":
        num_beeps = 2
    elif spec.pattern == "triple":
        num_beeps = 3
    elif spec.pattern == "continuous":
        num_beeps = 5  # 5 beeps for continuous pattern
    else:
        raise ValueError(f"Unknown pattern: {spec.pattern}")

    # Generate single beep
    beep_samples = int(spec.duration * sample_rate)
    t = np.linspace(0, spec.duration, beep_samples, endpoint=False)
    beep = np.sin(2 * np.pi * spec.frequency * t)

    # Apply envelope
    beep = _apply_adsr_envelope(beep, sample_rate, spec.envelope)

    # Apply volume
    beep *= spec.volume

    # Create pattern with gaps
    if num_beeps == 1:
        audio = beep
    else:
        gap_samples = int(spec.gap_duration * sample_rate)
        gap = np.zeros(gap_samples)

        # Concatenate beeps with gaps
        segments = []
        for i in range(num_beeps):
            segments.append(beep)
            if i < num_beeps - 1:  # No gap after last beep
                segments.append(gap)

        audio = np.concatenate(segments)

    # Normalize to prevent clipping (with headroom)
    max_amplitude = np.abs(audio).max()
    if max_amplitude > 0:
        audio = audio / max_amplitude * 0.9

    return audio.astype(np.float32)


def synthesize_two_tone_alert(
    frequency1: float,
    frequency2: float,
    duration: float = 0.3,
    position: Tuple[float, float, float] = (0, 0, 0),
    volume: float = 1.0,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Generate two-tone alert (e.g., C5 + E5 chime).

    Args:
        frequency1: First tone frequency in Hz
        frequency2: Second tone frequency in Hz
        duration: Duration of each tone in seconds
        position: 3D position (x, y, z) in meters
        volume: Relative loudness (0.0-1.0)
        sample_rate: Sample rate in Hz

    Returns:
        Mono audio with two sequential tones
    """
    # Generate first tone
    spec1 = AlertSoundSpec(
        alert_type="custom",
        frequency=frequency1,
        duration=duration,
        pattern="single",
        envelope="soft",
        position=position,
        volume=volume,
    )
    tone1 = synthesize_alert_sound(spec1, sample_rate)

    # Generate second tone
    spec2 = AlertSoundSpec(
        alert_type="custom",
        frequency=frequency2,
        duration=duration,
        pattern="single",
        envelope="soft",
        position=position,
        volume=volume,
    )
    tone2 = synthesize_alert_sound(spec2, sample_rate)

    # Concatenate with small gap
    gap = np.zeros(int(0.05 * sample_rate))
    audio = np.concatenate([tone1, gap, tone2])

    return audio.astype(np.float32)


def place_alert_in_timeline(
    alert_audio: np.ndarray,
    trigger_time: float,
    total_duration: float,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Place alert sound at specific time in timeline.

    Args:
        alert_audio: Alert audio to place
        trigger_time: When to trigger alert (seconds)
        total_duration: Total timeline duration (seconds)
        sample_rate: Sample rate in Hz

    Returns:
        Audio array with alert placed at trigger_time, padded to total_duration
    """
    total_samples = int(total_duration * sample_rate)
    trigger_sample = int(trigger_time * sample_rate)

    # Create timeline
    timeline = np.zeros(total_samples, dtype=np.float32)

    # Place alert
    alert_len = len(alert_audio)
    end_sample = min(trigger_sample + alert_len, total_samples)
    actual_len = end_sample - trigger_sample

    if actual_len > 0:
        timeline[trigger_sample:end_sample] = alert_audio[:actual_len]

    return timeline


# Example usage
if __name__ == "__main__":
    import soundfile as sf

    print("Alert Sound Synthesis Examples")
    print("=" * 60)

    # Example 1: All preset types
    print("\nGenerating all preset alert types...")
    for alert_type in ALERT_PRESETS.keys():
        spec = create_preset_alert(alert_type, position=(0, 0, 0))
        audio = synthesize_alert_sound(spec)
        filename = f"outputs/test_alert_{alert_type}.wav"
        sf.write(filename, audio, 44100)
        print(f"  {alert_type}: {filename} ({len(audio)} samples)")

    # Example 2: Two-tone chime (C5 + E5)
    print("\nGenerating two-tone chime (C5 + E5)...")
    chime = synthesize_two_tone_alert(
        frequency1=523.25,  # C5
        frequency2=659.25,  # E5
        duration=0.3,
        volume=0.8
    )
    sf.write("outputs/test_alert_two_tone.wav", chime, 44100)
    print(f"  Saved to outputs/test_alert_two_tone.wav ({len(chime)} samples)")

    # Example 3: Alert in timeline (beep at 1.5 seconds)
    print("\nGenerating alert in timeline (beep at 1.5s)...")
    beep_spec = create_preset_alert("double_beep", position=(0, 0, 0))
    beep = synthesize_alert_sound(beep_spec)
    timeline = place_alert_in_timeline(beep, trigger_time=1.5, total_duration=3.0)
    sf.write("outputs/test_alert_timeline.wav", timeline, 44100)
    print(f"  Saved to outputs/test_alert_timeline.wav ({len(timeline)} samples)")

    # Example 4: Multiple alerts in timeline
    print("\nGenerating multiple alerts in timeline...")
    timeline = np.zeros(int(5.0 * 44100), dtype=np.float32)

    # Beep at 0.5s
    beep1 = synthesize_alert_sound(create_preset_alert("beep", (0, 0, 0)))
    timeline += place_alert_in_timeline(beep1, 0.5, 5.0)

    # Warning at 2.0s
    warning = synthesize_alert_sound(create_preset_alert("warning", (0, 0, 0)))
    timeline += place_alert_in_timeline(warning, 2.0, 5.0)

    # Error at 4.0s
    error = synthesize_alert_sound(create_preset_alert("error", (0, 0, 0)))
    timeline += place_alert_in_timeline(error, 4.0, 5.0)

    sf.write("outputs/test_alert_multiple.wav", timeline, 44100)
    print(f"  Saved to outputs/test_alert_multiple.wav ({len(timeline)} samples)")

    print("\nDone! Check outputs/ directory for generated audio files.")
