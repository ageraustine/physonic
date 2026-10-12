"""Pytest fixtures for spatial audio tests."""
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from physonic.spatial.hrtf import HRTFSet, load_kemar_hrtf
from physonic.spatial.microphone import BinauralMicrophone


@pytest.fixture(scope="session")
def hrtf_mini_path(tmp_path_factory):
    """Create a minimal HRTF dataset for testing."""
    hrtf_dir = tmp_path_factory.mktemp("hrtf_mini") / "compact"

    # Create a simple test dataset with 1 elevation, 5 azimuths
    sr = 44100
    hrir_length = 128

    elevation = 0
    azimuths = [0, 45, 90, 135, 180]

    elev_dir = hrtf_dir / f"elev{elevation}"
    elev_dir.mkdir(parents=True, exist_ok=True)

    for azim in azimuths:
        # Generate synthetic HRIRs with realistic ITD/ILD cues
        t = np.arange(hrir_length) / sr

        # ITD: delay for positive azimuths (left ear further from source)
        # Max ITD ~0.7ms at 90°
        itd_sec = (azim / 90.0) * 0.0007
        delay_samp_l = int(itd_sec * sr)

        # Create impulse responses with exponential decay
        hrir_l = np.zeros(hrir_length)
        hrir_r = np.zeros(hrir_length)

        # Right ear: onset at sample 10
        if 10 + 20 < hrir_length:
            hrir_r[10:10+20] = np.hanning(20) * 0.1

        # Left ear: delayed by ITD
        onset_l = 10 + delay_samp_l
        if onset_l + 20 < hrir_length:
            hrir_l[onset_l:onset_l+20] = np.hanning(20) * 0.1

        # ILD: attenuate shadowed ear (frequency-dependent in reality)
        # Simplified: attenuate based on azimuth
        shadow_factor = np.abs(np.cos(np.radians(azim)))
        hrir_r *= (1 + shadow_factor) / 2
        hrir_l *= (1 - shadow_factor) / 2 + 0.5  # Don't go to zero

        # Write WAV files
        sf.write(elev_dir / f"L{elevation}e{azim:03d}a.wav", hrir_l, sr)
        sf.write(elev_dir / f"R{elevation}e{azim:03d}a.wav", hrir_r, sr)

    return hrtf_dir.parent


@pytest.fixture(scope="session")
def hrtf_mini(hrtf_mini_path):
    """Load the minimal HRTF dataset."""
    return load_kemar_hrtf(str(hrtf_mini_path))


@pytest.fixture
def impulse_1sec():
    """1-second impulse signal at sample 100."""
    signal = np.zeros(44100)
    signal[100] = 1.0
    return signal


@pytest.fixture
def tone_4khz():
    """1-second 4 kHz tone for ILD testing."""
    sr = 44100
    duration = 0.1  # 100ms
    t = np.arange(int(duration * sr)) / sr
    return np.sin(2 * np.pi * 4000 * t)


@pytest.fixture
def binaural_mic():
    """Default binaural microphone at origin, facing +X."""
    return BinauralMicrophone(
        position=np.array([0.0, 0.0, 1.0]),
        orientation=0.0
    )
