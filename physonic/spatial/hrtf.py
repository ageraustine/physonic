"""HRTF loading, interpolation, and convolution for binaural audio rendering."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

try:
    import resampy
    HAS_RESAMPY = True
except ImportError:
    HAS_RESAMPY = False


@dataclass
class HRTFSet:
    """HRTF measurements for all measured angles.

    Attributes:
        sample_rate: Audio sample rate in Hz (typically 44100)
        elevations: Array of measured elevation angles in degrees
        azimuths: Dict mapping elevation to array of measured azimuths
        hrir_left: Dict mapping (elevation, azimuth) to left-ear impulse response
        hrir_right: Dict mapping (elevation, azimuth) to right-ear impulse response
        hrir_length: Length of HRIRs in samples (typically 128 or 256)
    """
    sample_rate: int
    elevations: np.ndarray
    azimuths: Dict[int, np.ndarray]
    hrir_left: Dict[Tuple[int, int], np.ndarray]
    hrir_right: Dict[Tuple[int, int], np.ndarray]
    hrir_length: int


def load_kemar_hrtf(dataset_path: str, target_sr: int = 44100, hrir_length: int = 128) -> HRTFSet:
    """Load MIT KEMAR HRTF dataset from disk.

    Expects MIT KEMAR "compact" directory structure:
        dataset_path/
            compact/
                elev0/
                    L0e000a.wav    # Left ear, 0° elevation, 0° azimuth
                    R0e000a.wav    # Right ear
                    L0e015a.wav    # 15° azimuth
                    ...
                elev-40/
                    ...

    Args:
        dataset_path: Path to HRTF root directory (contains 'compact/' subdirectory)
        target_sr: Target sample rate in Hz (will resample if needed)
        hrir_length: Target HRIR length in samples (will truncate or zero-pad)

    Returns:
        HRTFSet with loaded and processed impulse responses

    Raises:
        FileNotFoundError: If dataset_path doesn't exist or is missing expected structure
        ValueError: If WAV files have inconsistent sample rates or invalid format

    Example:
        >>> hrtf = load_kemar_hrtf("./assets/hrtf/mit_kemar/")
        >>> hrtf.sample_rate
        44100
        >>> hrtf.elevations
        array([-40, -30, ..., 80, 90])
        >>> hrtf.hrir_left[(0, 45)].shape
        (128,)
    """
    root = Path(dataset_path) / "compact"
    if not root.exists():
        raise FileNotFoundError(f"HRTF dataset path not found: {root}")

    elevations = []
    azimuths_by_elev = {}
    hrir_left = {}
    hrir_right = {}
    detected_sr = None

    # Scan elevation directories
    elev_dirs = sorted(root.glob("elev*"))
    if not elev_dirs:
        raise FileNotFoundError(f"No elevation directories found in {root}")

    for elev_dir in elev_dirs:
        # Parse elevation from directory name: "elev0" → 0, "elev-40" → -40
        elev_str = elev_dir.name.replace("elev", "")
        elev = int(elev_str)
        elevations.append(elev)

        azims = []

        # Load left/right ear pairs
        for wav_path in sorted(elev_dir.glob("L*e*a.wav")):
            # Parse azimuth: "L0e045a.wav" → azimuth 45
            parts = wav_path.stem  # "L0e045a"
            try:
                azim_str = parts.split('e')[1].replace('a', '')
                azim = int(azim_str)
            except (IndexError, ValueError):
                continue  # Skip malformed filenames

            azims.append(azim)

            # Load left and right HRIRs
            hrir_l, sr = sf.read(wav_path)
            wav_path_r = wav_path.parent / wav_path.name.replace('L', 'R')
            if not wav_path_r.exists():
                raise FileNotFoundError(f"Missing right-ear HRIR for {wav_path}")
            hrir_r, sr_r = sf.read(wav_path_r)

            # Verify sample rates match
            if sr != sr_r:
                raise ValueError(f"Sample rate mismatch: {wav_path} ({sr} Hz) vs {wav_path_r} ({sr_r} Hz)")

            if detected_sr is None:
                detected_sr = sr
            elif sr != detected_sr:
                raise ValueError(f"Inconsistent sample rates: found {sr} Hz and {detected_sr} Hz")

            # Resample if needed
            if sr != target_sr:
                if not HAS_RESAMPY:
                    raise ImportError("resampy required for sample rate conversion. Install with: pip install resampy")
                hrir_l = resampy.resample(hrir_l, sr, target_sr)
                hrir_r = resampy.resample(hrir_r, sr, target_sr)

            # Truncate or zero-pad to target length
            if len(hrir_l) > hrir_length:
                hrir_l = hrir_l[:hrir_length]
                hrir_r = hrir_r[:hrir_length]
            elif len(hrir_l) < hrir_length:
                hrir_l = np.pad(hrir_l, (0, hrir_length - len(hrir_l)))
                hrir_r = np.pad(hrir_r, (0, hrir_length - len(hrir_r)))

            # Store
            hrir_left[(elev, azim)] = hrir_l.astype(np.float32)
            hrir_right[(elev, azim)] = hrir_r.astype(np.float32)

        if azims:
            azimuths_by_elev[elev] = np.array(sorted(azims))

    if not elevations:
        raise ValueError(f"No valid HRTF measurements found in {root}")

    return HRTFSet(
        sample_rate=target_sr,
        elevations=np.array(sorted(elevations)),
        azimuths=azimuths_by_elev,
        hrir_left=hrir_left,
        hrir_right=hrir_right,
        hrir_length=hrir_length
    )


def interpolate_hrtf(
    hrtf: HRTFSet,
    azimuth: float,
    elevation: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Get HRTF impulse responses for arbitrary angle using nearest-neighbor interpolation.

    Phase 2A uses simple nearest-neighbor interpolation. Future versions may implement
    VBAP (Vector Base Amplitude Panning) for smoother interpolation.

    Args:
        hrtf: Loaded HRTF dataset
        azimuth: Azimuth angle in degrees, range [-180, 180]
                 (0 = front, 90 = left, -90 = right, 180 = back)
        elevation: Elevation angle in degrees, range depends on dataset
                   (0 = horizontal, positive = above, negative = below)

    Returns:
        (hrir_left, hrir_right): Tuple of impulse responses, each shape (hrtf.hrir_length,)

    Raises:
        ValueError: If elevation is outside the dataset's measured range

    Example:
        >>> hrir_l, hrir_r = interpolate_hrtf(hrtf, azimuth=45, elevation=0)
        >>> hrir_l.shape
        (128,)
        >>> # For intermediate angles, returns nearest measured angle
        >>> hrir_l_47, _ = interpolate_hrtf(hrtf, azimuth=47, elevation=0)
        >>> # Should return same as 45° if that's the nearest measurement
    """
    # Snap elevation to nearest measured value
    elev_idx = np.argmin(np.abs(hrtf.elevations - elevation))
    elev_nearest = int(hrtf.elevations[elev_idx])

    if elev_nearest not in hrtf.azimuths:
        raise ValueError(f"Elevation {elev_nearest}° not found in dataset. "
                        f"Available: {sorted(hrtf.azimuths.keys())}")

    # Snap azimuth to nearest measured value for this elevation
    available_azims = hrtf.azimuths[elev_nearest]
    azim_idx = np.argmin(np.abs(available_azims - azimuth))
    azim_nearest = int(available_azims[azim_idx])

    # Retrieve HRIRs
    key = (elev_nearest, azim_nearest)
    if key not in hrtf.hrir_left:
        raise ValueError(f"HRIR not found for (elev={elev_nearest}°, azim={azim_nearest}°)")

    return hrtf.hrir_left[key], hrtf.hrir_right[key]


def convolve_hrtf(
    mono: np.ndarray,
    hrir_left: np.ndarray,
    hrir_right: np.ndarray
) -> np.ndarray:
    """Apply HRTF filtering to mono signal using FFT-based convolution.

    Args:
        mono: Mono audio signal, shape (N,)
        hrir_left: Left-ear impulse response, shape (M,)
        hrir_right: Right-ear impulse response, shape (M,)

    Returns:
        Stereo audio, shape (N + M - 1, 2) where columns are [left, right]

    Note:
        Uses scipy.signal.fftconvolve for efficiency with longer HRIRs.
        Output length is N + M - 1 (full convolution).

    Example:
        >>> mono = np.random.randn(44100)  # 1 second
        >>> hrir_l, hrir_r = interpolate_hrtf(hrtf, azimuth=90, elevation=0)
        >>> stereo = convolve_hrtf(mono, hrir_l, hrir_r)
        >>> stereo.shape
        (44228, 2)  # 44100 + 128 - 1
        >>> stereo[:, 0]  # Left channel
        >>> stereo[:, 1]  # Right channel
    """
    # Convolve mono signal with left and right HRIRs
    left = fftconvolve(mono, hrir_left, mode='full')
    right = fftconvolve(mono, hrir_right, mode='full')

    # Stack into stereo array
    stereo = np.stack([left, right], axis=1)

    return stereo.astype(np.float32)
