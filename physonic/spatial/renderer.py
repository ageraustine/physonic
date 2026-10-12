"""Spatial audio rendering: combine distance attenuation with HRTF filtering."""
import numpy as np
from physonic.spatial.hrtf import HRTFSet, interpolate_hrtf, convolve_hrtf
from physonic.spatial.microphone import BinauralMicrophone


def render_spatial_source(
    mono: np.ndarray,
    source_position: np.ndarray,
    microphone: BinauralMicrophone,
    hrtf: HRTFSet
) -> np.ndarray:
    """Render mono audio source as binaural stereo from microphone perspective.

    Applies:
        1. Distance attenuation (1/r law)
        2. HRTF filtering based on source angle relative to microphone

    Args:
        mono: Mono audio from source, shape (N,)
        source_position: Source 3D position [x, y, z] in meters
        microphone: Binaural microphone with position and orientation
        hrtf: Loaded HRTF dataset

    Returns:
        Binaural stereo audio, shape (N + hrtf.hrir_length - 1, 2)
        Columns are [left_channel, right_channel]

    Example:
        >>> hrtf = load_kemar_hrtf("assets/hrtf/mit_kemar/")
        >>> mono = np.random.randn(44100)  # 1 sec impact sound
        >>> source_pos = np.array([2, 1, 0.5])  # 2m front, 1m left, 0.5m high
        >>> mic = BinauralMicrophone(position=np.array([0, 0, 1]), orientation=0)
        >>> stereo = render_spatial_source(mono, source_pos, mic, hrtf)
        >>> stereo.shape
        (44228, 2)  # 44100 + 128 - 1
        >>> stereo[:, 0]  # Left channel
        >>> stereo[:, 1]  # Right channel
    """
    # 1. Compute angle from microphone to source
    azimuth, elevation = microphone.compute_source_angle(source_position)

    # 2. Compute distance for attenuation
    distance = np.linalg.norm(source_position - microphone.position)
    # Clamp to minimum distance to avoid division by zero and excessive amplification
    distance = max(distance, 0.1)

    # 3. Apply distance attenuation (inverse distance law: amplitude ∝ 1/r)
    mono_attenuated = mono / distance

    # 4. Get HRTF for computed angle
    hrir_left, hrir_right = interpolate_hrtf(hrtf, azimuth, elevation)

    # 5. Apply HRTF convolution
    stereo = convolve_hrtf(mono_attenuated, hrir_left, hrir_right)

    return stereo
