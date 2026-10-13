"""Enhanced spatial audio rendering with complete acoustic physics.

Implements all proper acoustic propagation effects:
- Distance attenuation (inverse square law)
- Doppler effect (frequency shift from motion)
- Air absorption (frequency-dependent attenuation)
- Propagation delay (speed of sound)
- HRTF filtering (directional hearing)
- Occlusion (optional, requires line-of-sight info)
"""

import numpy as np
from typing import Optional, Tuple
from physonic.spatial.hrtf import HRTFSet, interpolate_hrtf, convolve_hrtf
from physonic.spatial.microphone import BinauralMicrophone
from physonic.spatial.physics import (
    calculate_doppler_shift,
    apply_doppler_shift,
    apply_air_absorption,
    calculate_propagation_delay,
    apply_propagation_delay,
    apply_occlusion,
)


def render_spatial_source_enhanced(
    mono: np.ndarray,
    source_position: np.ndarray,
    microphone: BinauralMicrophone,
    hrtf: HRTFSet,
    receiver_velocity: Optional[np.ndarray] = None,
    source_velocity: Optional[np.ndarray] = None,
    apply_doppler: bool = True,
    apply_absorption: bool = True,
    apply_delay: bool = True,
    occlusion_factor: float = 0.0,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Render mono audio with complete acoustic physics.

    Applies (in order):
        1. Doppler effect (frequency shift from motion)
        2. Air absorption (distance + frequency-dependent attenuation)
        3. Occlusion (if objects block line-of-sight)
        4. Propagation delay (speed of sound)
        5. HRTF filtering (directional perception)

    Args:
        mono: Mono audio from source, shape (N,)
        source_position: Source 3D position [x, y, z] in meters
        microphone: Binaural microphone with position and orientation
        hrtf: Loaded HRTF dataset
        receiver_velocity: Receiver velocity [vx, vy, vz] in m/s (for Doppler)
        source_velocity: Source velocity in m/s (default: stationary)
        apply_doppler: Enable Doppler effect
        apply_absorption: Enable air absorption
        apply_delay: Enable propagation delay
        occlusion_factor: Occlusion amount (0.0-1.0, 0 = clear path)
        sample_rate: Audio sample rate in Hz

    Returns:
        Binaural stereo audio, shape (N_out, 2) where N_out >= N
        Columns are [left_channel, right_channel]

    Example:
        >>> # Robot moving toward server at 2 m/s
        >>> hrtf = load_kemar_hrtf("assets/hrtf/test_kemar/")
        >>> mono = synthesize_server_hum(duration=0.1)  # 0.1s chunk
        >>> source_pos = np.array([5, 0, 1])
        >>> mic = BinauralMicrophone(position=np.array([0, 0, 1]), orientation=0)
        >>> receiver_vel = np.array([2, 0, 0])  # Moving toward source
        >>> stereo = render_spatial_source_enhanced(
        ...     mono, source_pos, mic, hrtf,
        ...     receiver_velocity=receiver_vel
        ... )
        >>> # stereo contains:
        >>> #   - Slightly higher frequency (Doppler)
        >>> #   - Attenuated by distance + air
        >>> #   - Delayed by ~14.5ms (5m / 343 m/s)
        >>> #   - Spatialized with HRTF
    """
    # Calculate distance and direction
    relative_position = source_position - microphone.position
    distance = np.linalg.norm(relative_position)
    distance = max(distance, 0.1)  # Avoid division by zero

    # Start with original mono audio
    processed = mono.copy()

    # 1. Apply Doppler effect (if receiver is moving)
    if apply_doppler and receiver_velocity is not None:
        doppler_ratio = calculate_doppler_shift(
            source_position,
            microphone.position,
            receiver_velocity,
            source_velocity,
        )

        # Only apply if shift is noticeable (>0.1%)
        if abs(doppler_ratio - 1.0) > 0.001:
            processed = apply_doppler_shift(processed, doppler_ratio, sample_rate)

    # 2. Apply air absorption (distance + frequency-dependent)
    if apply_absorption:
        processed = apply_air_absorption(processed, distance, sample_rate)

    # 3. Apply occlusion (if objects block path)
    if occlusion_factor > 0.0:
        processed = apply_occlusion(processed, occlusion_factor, sample_rate)

    # 4. Compute azimuth and elevation for HRTF
    azimuth, elevation = microphone.compute_source_angle(source_position)

    # 5. Get HRTF filters
    hrir_left, hrir_right = interpolate_hrtf(hrtf, azimuth, elevation)

    # 6. Apply HRTF convolution
    stereo = convolve_hrtf(processed, hrir_left, hrir_right)

    # 7. Apply propagation delay (after HRTF to preserve stereo)
    if apply_delay:
        delay_samples = calculate_propagation_delay(distance, sample_rate=sample_rate)
        if delay_samples > 0:
            stereo_delayed = np.zeros((len(stereo) + delay_samples, 2), dtype=np.float32)
            stereo_delayed[delay_samples:] = stereo
            stereo = stereo_delayed

    return stereo


def render_spatial_source(
    mono: np.ndarray,
    source_position: np.ndarray,
    microphone: BinauralMicrophone,
    hrtf: HRTFSet,
) -> np.ndarray:
    """Simple spatial rendering (backwards compatibility).

    Applies only distance attenuation and HRTF (no Doppler, absorption, delay).

    For full physics, use render_spatial_source_enhanced().

    Args:
        mono: Mono audio from source
        source_position: Source 3D position
        microphone: Binaural microphone
        hrtf: HRTF dataset

    Returns:
        Binaural stereo audio
    """
    # Fallback to simple rendering (original behavior)
    distance = np.linalg.norm(source_position - microphone.position)
    distance = max(distance, 0.1)

    # Apply distance attenuation only
    mono_attenuated = mono / distance

    # Apply HRTF
    azimuth, elevation = microphone.compute_source_angle(source_position)
    hrir_left, hrir_right = interpolate_hrtf(hrtf, azimuth, elevation)
    stereo = convolve_hrtf(mono_attenuated, hrir_left, hrir_right)

    return stereo
