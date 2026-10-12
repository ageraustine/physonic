"""Audio feature extraction for acoustic perception tasks."""

import numpy as np
import librosa


def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 44100,
    n_mels: int = 64,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 20.0,
    fmax: float = 8000.0
) -> np.ndarray:
    """
    Extract mel-spectrogram from binaural audio.

    Args:
        audio: Stereo audio (T, 2)
        sr: Sample rate
        n_mels: Number of mel bins
        n_fft: FFT window size
        hop_length: Hop length for STFT
        fmin: Minimum frequency
        fmax: Maximum frequency

    Returns:
        Mel-spectrogram (n_mels, time_frames, 2) in dB scale
    """
    left = audio[:, 0]
    right = audio[:, 1]

    # Compute mel-spectrogram for each channel
    mel_left = librosa.feature.melspectrogram(
        y=left, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmin=fmin, fmax=fmax
    )
    mel_right = librosa.feature.melspectrogram(
        y=right, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmin=fmin, fmax=fmax
    )

    # Convert to dB scale
    mel_left_db = librosa.power_to_db(mel_left, ref=np.max)
    mel_right_db = librosa.power_to_db(mel_right, ref=np.max)

    # Stack: (n_mels, time_frames, 2)
    mel_spec = np.stack([mel_left_db, mel_right_db], axis=-1)

    return mel_spec.astype(np.float32)


def extract_mfcc(
    audio: np.ndarray,
    sr: int = 44100,
    n_mfcc: int = 13
) -> np.ndarray:
    """Extract MFCC features from binaural audio."""
    left = audio[:, 0]
    right = audio[:, 1]

    mfcc_left = librosa.feature.mfcc(y=left, sr=sr, n_mfcc=n_mfcc)
    mfcc_right = librosa.feature.mfcc(y=right, sr=sr, n_mfcc=n_mfcc)

    # (n_mfcc, time_frames, 2)
    mfcc = np.stack([mfcc_left, mfcc_right], axis=-1)
    return mfcc.astype(np.float32)
