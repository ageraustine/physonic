"""Tests for audio preprocessing functions."""

import pytest
import numpy as np

from physonic.envs.preprocessing import extract_mel_spectrogram, extract_mfcc


class TestMelSpectrogramExtraction:
    """Test mel-spectrogram extraction."""

    def test_output_shape(self):
        """Test mel-spectrogram has correct shape."""
        # 1 second stereo audio @ 44.1kHz
        audio = np.random.randn(44100, 2).astype(np.float32)

        mel = extract_mel_spectrogram(audio, sr=44100, n_mels=64)

        # Expected shape: (n_mels, time_frames, 2)
        # time_frames ≈ ceil(44100 / 512) ≈ 87
        assert mel.shape[0] == 64
        assert mel.shape[2] == 2
        assert 85 <= mel.shape[1] <= 90  # Approximate time frames

    def test_output_dtype(self):
        """Test mel-spectrogram is float32."""
        audio = np.random.randn(44100, 2).astype(np.float32)
        mel = extract_mel_spectrogram(audio)
        assert mel.dtype == np.float32

    def test_db_scale_range(self):
        """Test mel-spectrogram values are in dB scale."""
        audio = np.random.randn(44100, 2).astype(np.float32) * 0.1

        mel = extract_mel_spectrogram(audio)

        # dB scale should be negative (relative to max)
        # Most values should be below 0 dB
        assert np.all(mel <= 0.1)  # Small tolerance for numerical issues
        assert np.any(mel < -10.0)  # Should have some low values

    def test_custom_parameters(self):
        """Test mel-spectrogram with custom parameters."""
        audio = np.random.randn(22050, 2).astype(np.float32)  # 0.5s @ 44.1kHz

        mel = extract_mel_spectrogram(
            audio, sr=44100, n_mels=128, n_fft=4096, hop_length=1024
        )

        assert mel.shape[0] == 128
        # time_frames ≈ ceil(22050 / 1024) ≈ 22
        assert 20 <= mel.shape[1] <= 25

    def test_channel_independence(self):
        """Test left and right channels are processed independently."""
        # Create audio with different left and right channels
        left = np.random.randn(44100)
        right = np.zeros(44100)  # Silent right channel
        audio = np.stack([left, right], axis=-1).astype(np.float32)

        mel = extract_mel_spectrogram(audio)

        # Left channel (index 0) should have more energy
        left_energy = np.mean(np.abs(mel[:, :, 0]))
        right_energy = np.mean(np.abs(mel[:, :, 1]))

        assert left_energy > right_energy


class TestMFCCExtraction:
    """Test MFCC extraction."""

    def test_output_shape(self):
        """Test MFCC has correct shape."""
        audio = np.random.randn(44100, 2).astype(np.float32)

        mfcc = extract_mfcc(audio, sr=44100, n_mfcc=13)

        # Expected shape: (n_mfcc, time_frames, 2)
        assert mfcc.shape[0] == 13
        assert mfcc.shape[2] == 2
        assert mfcc.shape[1] > 0  # time_frames depends on default hop_length

    def test_output_dtype(self):
        """Test MFCC is float32."""
        audio = np.random.randn(44100, 2).astype(np.float32)
        mfcc = extract_mfcc(audio)
        assert mfcc.dtype == np.float32

    def test_custom_n_mfcc(self):
        """Test MFCC with custom number of coefficients."""
        audio = np.random.randn(44100, 2).astype(np.float32)

        mfcc = extract_mfcc(audio, n_mfcc=20)

        assert mfcc.shape[0] == 20
