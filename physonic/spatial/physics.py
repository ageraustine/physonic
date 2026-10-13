"""Physical acoustic propagation models.

Implements proper acoustic physics:
- Doppler effect (frequency shift from motion)
- Air absorption (frequency-dependent attenuation)
- Propagation delay (speed of sound)
- Occlusion (objects blocking sound paths)
"""

import numpy as np
import scipy.signal
from typing import Tuple, Optional


# Physical constants
SPEED_OF_SOUND = 343.0  # m/s at 20°C, sea level
REFERENCE_DISTANCE = 1.0  # Reference distance for inverse square law (meters)

# Air absorption coefficients (dB/m) at 20°C, 50% humidity
# Source: ISO 9613-1
AIR_ABSORPTION_COEFFICIENTS = {
    63: 0.00011,
    125: 0.00041,
    250: 0.0015,
    500: 0.0046,
    1000: 0.0117,
    2000: 0.0348,
    4000: 0.1090,
    8000: 0.3630,
}


def calculate_doppler_shift(
    source_position: np.ndarray,
    receiver_position: np.ndarray,
    receiver_velocity: np.ndarray,
    source_velocity: np.ndarray = None,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> float:
    """Calculate Doppler frequency shift ratio.

    Args:
        source_position: Source 3D position [x, y, z] in meters
        receiver_position: Receiver 3D position [x, y, z] in meters
        receiver_velocity: Receiver 3D velocity [vx, vy, vz] in m/s
        source_velocity: Source 3D velocity (default: stationary)
        speed_of_sound: Speed of sound in m/s (default: 343 m/s)

    Returns:
        Frequency ratio: f_observed / f_source
        - > 1.0: frequencies increase (approaching)
        - < 1.0: frequencies decrease (receding)
        - = 1.0: no relative motion

    Example:
        >>> source_pos = np.array([10, 0, 0])
        >>> receiver_pos = np.array([0, 0, 0])
        >>> receiver_vel = np.array([3, 0, 0])  # Moving toward source at 3 m/s
        >>> ratio = calculate_doppler_shift(source_pos, receiver_pos, receiver_vel)
        >>> ratio
        1.0087  # ~0.87% frequency increase
    """
    if source_velocity is None:
        source_velocity = np.zeros(3)

    # Vector from receiver to source
    relative_position = source_position - receiver_position
    distance = np.linalg.norm(relative_position)

    if distance < 1e-6:
        return 1.0  # No Doppler at same position

    # Unit vector pointing from receiver to source
    direction = relative_position / distance

    # Radial velocities (positive = moving apart)
    v_receiver_radial = np.dot(receiver_velocity, direction)
    v_source_radial = np.dot(source_velocity, direction)

    # Doppler formula: f_observed / f_source = (c - v_receiver) / (c - v_source)
    # Where positive velocity = moving toward each other
    # We negate v_receiver because our convention is velocity of receiver,
    # but formula uses velocity toward source
    numerator = speed_of_sound + v_receiver_radial
    denominator = speed_of_sound + v_source_radial

    # Clamp to avoid unrealistic values
    if abs(denominator) < 1e-6:
        return 1.0

    doppler_ratio = numerator / denominator

    # Clamp to reasonable range (±50% shift)
    doppler_ratio = np.clip(doppler_ratio, 0.5, 1.5)

    return doppler_ratio


def apply_doppler_shift(
    audio: np.ndarray,
    doppler_ratio: float,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Apply Doppler frequency shift to audio signal.

    Uses resampling to shift all frequencies by the Doppler ratio.

    Args:
        audio: Input audio signal, shape (N,)
        doppler_ratio: Frequency ratio (f_observed / f_source)
        sample_rate: Sample rate in Hz

    Returns:
        Doppler-shifted audio, same length as input

    Example:
        >>> # Simulate approaching sound source (frequencies increase)
        >>> original = synthesize_tone(440, duration=1.0, sample_rate=44100)
        >>> shifted = apply_doppler_shift(original, doppler_ratio=1.01)
        >>> # shifted now contains 444.4 Hz tone (1% higher)
    """
    if abs(doppler_ratio - 1.0) < 1e-6:
        return audio  # No shift needed

    # Resample to apply frequency shift
    # If doppler_ratio > 1.0: frequencies increase (compress time)
    # If doppler_ratio < 1.0: frequencies decrease (stretch time)
    num_samples = len(audio)
    resampled_length = int(num_samples / doppler_ratio)

    # Resample using scipy
    resampled = scipy.signal.resample(audio, resampled_length)

    # Trim or pad to original length
    if len(resampled) > num_samples:
        return resampled[:num_samples]
    elif len(resampled) < num_samples:
        return np.pad(resampled, (0, num_samples - len(resampled)), mode='constant')
    else:
        return resampled


def calculate_air_absorption_filter(
    distance: float,
    sample_rate: int = 44100,
    filter_order: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Design frequency-dependent air absorption filter.

    Air absorbs high frequencies more than low frequencies.
    Returns filter coefficients for scipy.signal.sosfilt.

    Args:
        distance: Propagation distance in meters
        sample_rate: Sample rate in Hz
        filter_order: Filter order (higher = more accurate but slower)

    Returns:
        (sos_sections, gain): Second-order sections and gain factor
        - sos_sections: Filter coefficients for sosfilt
        - gain: Overall amplitude scaling (inverse square law)

    Example:
        >>> sos, gain = calculate_air_absorption_filter(distance=10.0)
        >>> filtered_audio = scipy.signal.sosfilt(sos, audio) * gain
    """
    # Inverse square law attenuation
    distance = max(distance, REFERENCE_DISTANCE)
    geometric_gain = REFERENCE_DISTANCE / distance

    # Frequency-dependent absorption
    # Design multi-band filter approximating air absorption curve
    nyquist = sample_rate / 2

    # Get absorption values at key frequencies
    frequencies = np.array(sorted(AIR_ABSORPTION_COEFFICIENTS.keys()), dtype=float)
    attenuations_db = np.array([AIR_ABSORPTION_COEFFICIENTS[int(f)] * distance for f in frequencies])

    # Convert dB to linear amplitude
    attenuations_linear = 10 ** (-attenuations_db / 20)

    # Design filter to match this frequency response
    # Use least-squares FIR filter design
    num_taps = filter_order * 8 + 1

    # Normalize frequencies to Nyquist
    freq_normalized = frequencies / nyquist

    # Only include frequencies below Nyquist
    valid_idx = freq_normalized < 1.0
    freq_normalized = freq_normalized[valid_idx]
    attenuations_linear = attenuations_linear[valid_idx]

    if len(freq_normalized) < 2:
        # No filtering needed
        return np.array([[[1, 0, 0, 1, 0, 0]]]), geometric_gain

    # Design FIR filter
    try:
        fir_coeffs = scipy.signal.firls(
            num_taps,
            freq_normalized,
            attenuations_linear,
            fs=2.0,  # Normalized frequency
        )

        # Convert FIR to SOS for numerical stability
        sos = scipy.signal.tf2sos([fir_coeffs], [1.0])

    except:
        # Fallback: simple low-pass filter
        cutoff = 4000.0 / nyquist  # Rolloff above 4kHz
        sos = scipy.signal.butter(4, cutoff, btype='low', output='sos', fs=2.0)

    return sos, geometric_gain


def apply_air_absorption(
    audio: np.ndarray,
    distance: float,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Apply air absorption to audio signal.

    Combines inverse square law and frequency-dependent absorption.

    Args:
        audio: Input audio signal, shape (N,)
        distance: Propagation distance in meters
        sample_rate: Sample rate in Hz

    Returns:
        Absorbed audio with distance attenuation and high-frequency rolloff

    Example:
        >>> # 10m distance: -20dB overall, high frequencies attenuated more
        >>> absorbed = apply_air_absorption(audio, distance=10.0)
    """
    if distance < 1e-3:
        return audio

    # Get filter coefficients
    sos, gain = calculate_air_absorption_filter(distance, sample_rate)

    # Apply filter
    filtered = scipy.signal.sosfilt(sos, audio)

    # Apply geometric attenuation
    return filtered * gain


def calculate_propagation_delay(
    distance: float,
    speed_of_sound: float = SPEED_OF_SOUND,
    sample_rate: int = 44100,
) -> int:
    """Calculate propagation delay in samples.

    Args:
        distance: Propagation distance in meters
        speed_of_sound: Speed of sound in m/s
        sample_rate: Sample rate in Hz

    Returns:
        Delay in samples (integer)

    Example:
        >>> # 10m distance = ~29ms = 1280 samples at 44.1kHz
        >>> delay = calculate_propagation_delay(10.0)
        >>> delay
        1280
    """
    delay_seconds = distance / speed_of_sound
    delay_samples = int(delay_seconds * sample_rate)
    return max(0, delay_samples)


def apply_propagation_delay(
    audio: np.ndarray,
    delay_samples: int,
) -> np.ndarray:
    """Apply propagation delay to audio signal.

    Args:
        audio: Input audio signal, shape (N,)
        delay_samples: Delay in samples

    Returns:
        Delayed audio (same length, zero-padded at start)

    Example:
        >>> delayed = apply_propagation_delay(audio, delay_samples=1000)
        >>> # First 1000 samples are zero, rest is original audio
    """
    if delay_samples <= 0:
        return audio

    # Shift audio forward in time
    delayed = np.zeros_like(audio)
    if delay_samples < len(audio):
        delayed[delay_samples:] = audio[:-delay_samples]

    return delayed


def apply_occlusion(
    audio: np.ndarray,
    occlusion_factor: float,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Apply occlusion (muffling) to audio signal.

    Simulates sound passing through/around obstacles.

    Args:
        audio: Input audio signal, shape (N,)
        occlusion_factor: Occlusion amount (0.0 = no occlusion, 1.0 = fully occluded)
        sample_rate: Sample rate in Hz

    Returns:
        Occluded audio (low-pass filtered + attenuated)

    Example:
        >>> # 50% occluded: -6dB, high frequencies muffled
        >>> occluded = apply_occlusion(audio, occlusion_factor=0.5)
    """
    if occlusion_factor < 1e-3:
        return audio

    occlusion_factor = np.clip(occlusion_factor, 0.0, 1.0)

    # Attenuation (dB loss proportional to occlusion)
    attenuation_db = -20 * occlusion_factor  # 0 to -20 dB
    gain = 10 ** (attenuation_db / 20)

    # Low-pass filter (cutoff inversely proportional to occlusion)
    nyquist = sample_rate / 2
    cutoff_freq = 8000 * (1.0 - occlusion_factor * 0.9)  # 8kHz to 800Hz
    cutoff_normalized = cutoff_freq / nyquist

    # Design low-pass filter
    sos = scipy.signal.butter(4, cutoff_normalized, btype='low', output='sos')

    # Apply filter and gain
    filtered = scipy.signal.sosfilt(sos, audio)
    return filtered * gain
