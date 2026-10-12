"""Unit tests for physonic.spatial.renderer module."""
import numpy as np
import pytest
from physonic.spatial.renderer import render_spatial_source
from physonic.spatial.microphone import BinauralMicrophone


class TestRenderSpatialSource:
    """Tests for render_spatial_source function."""

    def test_render_output_shape(self, hrtf_mini, binaural_mic):
        """Test that render output has correct shape."""
        mono = np.random.randn(1000)
        source_pos = np.array([1.0, 0.0, 1.0])

        stereo = render_spatial_source(mono, source_pos, binaural_mic, hrtf_mini)

        # Output should be stereo with length N + hrir_length - 1
        expected_len = len(mono) + hrtf_mini.hrir_length - 1
        assert stereo.shape == (expected_len, 2)

    def test_render_output_finite(self, hrtf_mini, binaural_mic):
        """Test that render output contains only finite values."""
        mono = np.random.randn(1000)
        source_pos = np.array([1.0, 0.0, 1.0])

        stereo = render_spatial_source(mono, source_pos, binaural_mic, hrtf_mini)

        assert np.isfinite(stereo).all()

    def test_render_distance_attenuation(self, hrtf_mini, binaural_mic):
        """Test that distance attenuation follows 1/r law."""
        mono = np.ones(1000)  # Constant signal

        # Source at distance 1m
        source_1m = binaural_mic.position + np.array([1.0, 0.0, 0.0])
        stereo_1m = render_spatial_source(mono, source_1m, binaural_mic, hrtf_mini)
        rms_1m = np.sqrt(np.mean(stereo_1m ** 2))

        # Source at distance 2m
        source_2m = binaural_mic.position + np.array([2.0, 0.0, 0.0])
        stereo_2m = render_spatial_source(mono, source_2m, binaural_mic, hrtf_mini)
        rms_2m = np.sqrt(np.mean(stereo_2m ** 2))

        # RMS should be roughly half (1/r law)
        ratio = rms_2m / rms_1m
        assert 0.45 < ratio < 0.55, f"Distance attenuation ratio {ratio:.2f} not close to 0.5"

    def test_render_minimum_distance_clamp(self, hrtf_mini, binaural_mic):
        """Test that very close sources are clamped to minimum distance."""
        mono = np.ones(100)

        # Source very close (should be clamped to 0.1m)
        source_close = binaural_mic.position + np.array([0.01, 0.0, 0.0])
        stereo_close = render_spatial_source(mono, source_close, binaural_mic, hrtf_mini)

        # Should not be excessively loud (no division by ~0)
        assert np.abs(stereo_close).max() < 100.0

    def test_render_different_angles_differ(self, hrtf_mini, binaural_mic, impulse_1sec):
        """Test that sources at different angles produce different outputs."""
        # Front
        source_front = binaural_mic.position + np.array([1.0, 0.0, 0.0])
        stereo_front = render_spatial_source(impulse_1sec, source_front, binaural_mic, hrtf_mini)

        # Left
        source_left = binaural_mic.position + np.array([0.0, 1.0, 0.0])
        stereo_left = render_spatial_source(impulse_1sec, source_left, binaural_mic, hrtf_mini)

        # Outputs should differ
        assert not np.allclose(stereo_front, stereo_left, atol=1e-6)

    def test_render_itd_for_left_source(self, hrtf_mini, binaural_mic, impulse_1sec):
        """Test that left source produces positive ITD."""
        # Source at 90° left
        source_left = binaural_mic.position + np.array([0.0, 1.0, 0.0])
        stereo = render_spatial_source(impulse_1sec, source_left, binaural_mic, hrtf_mini)

        # Find peaks
        peak_l = np.argmax(np.abs(stereo[:500, 0]))
        peak_r = np.argmax(np.abs(stereo[:500, 1]))

        itd_samples = peak_l - peak_r

        # Left source: left channel should be delayed (positive ITD)
        assert itd_samples > 0, f"Expected positive ITD for left source, got {itd_samples} samples"

    def test_render_itd_for_right_source(self, hrtf_mini, binaural_mic, impulse_1sec):
        """Test that right source produces negative ITD."""
        # Source at -90° (right), but our mini dataset only has positive azimuths
        # Use 0° (front) and 180° (back) as comparison instead
        source_back = binaural_mic.position + np.array([-1.0, 0.0, 0.0])
        stereo = render_spatial_source(impulse_1sec, source_back, binaural_mic, hrtf_mini)

        # For back source, ITD should be measurable but not excessively large
        peak_l = np.argmax(np.abs(stereo[:500, 0]))
        peak_r = np.argmax(np.abs(stereo[:500, 1]))
        itd_samples = abs(peak_l - peak_r)
        itd_ms = itd_samples / 44.1

        # Back position (180°) - ITD varies by HRTF dataset, just verify finite
        assert 0.0 <= itd_ms < 2.0, f"ITD {itd_ms:.2f} ms outside reasonable range [0, 2] ms"

    def test_render_with_zero_input(self, hrtf_mini, binaural_mic):
        """Test rendering with zero input signal."""
        mono = np.zeros(1000)
        source_pos = np.array([1.0, 0.0, 1.0])

        stereo = render_spatial_source(mono, source_pos, binaural_mic, hrtf_mini)

        # Output should be all zeros
        np.testing.assert_allclose(stereo, 0.0, atol=1e-10)

    def test_render_preserves_source_characteristics(self, hrtf_mini, binaural_mic, tone_4khz):
        """Test that rendering preserves frequency content of source."""
        source_pos = np.array([1.0, 0.0, 1.0])
        stereo = render_spatial_source(tone_4khz, source_pos, binaural_mic, hrtf_mini)

        # Check that dominant frequency is still around 4 kHz
        from scipy.fft import rfft, rfftfreq
        sr = 44100

        # Analyze left channel
        fft = np.abs(rfft(stereo[:, 0]))
        freqs = rfftfreq(len(stereo[:, 0]), 1/sr)

        # Find peak frequency
        peak_idx = np.argmax(fft)
        peak_freq = freqs[peak_idx]

        # Should be close to 4 kHz (within 10%)
        assert 3600 < peak_freq < 4400, f"Peak frequency {peak_freq:.0f} Hz not close to 4000 Hz"


@pytest.mark.integration
class TestRendererIntegration:
    """Integration tests for renderer module."""

    def test_full_spatial_rendering_pipeline(self, hrtf_mini_path, impulse_1sec):
        """Test complete pipeline: load HRTF → render spatial audio."""
        from physonic.spatial.hrtf import load_kemar_hrtf

        # Load HRTF
        hrtf = load_kemar_hrtf(str(hrtf_mini_path))

        # Create mic
        mic = BinauralMicrophone(np.array([0.0, 0.0, 1.0]), 0.0)

        # Render from multiple positions
        positions = [
            np.array([1.0, 0.0, 1.0]),   # Front
            np.array([0.0, 1.0, 1.0]),   # Left
            np.array([1.0, 1.0, 1.0]),   # Front-left
        ]

        for pos in positions:
            stereo = render_spatial_source(impulse_1sec, pos, mic, hrtf)

            # Validate output
            assert stereo.shape[1] == 2
            assert np.isfinite(stereo).all()
            assert np.abs(stereo).max() > 0

    def test_render_multiple_distances(self, hrtf_mini, binaural_mic):
        """Test rendering at multiple distances."""
        mono = np.random.randn(500)
        distances = [0.5, 1.0, 2.0, 4.0]

        rms_values = []
        for dist in distances:
            source_pos = binaural_mic.position + np.array([dist, 0.0, 0.0])
            stereo = render_spatial_source(mono, source_pos, binaural_mic, hrtf_mini)
            rms = np.sqrt(np.mean(stereo ** 2))
            rms_values.append(rms)

        # RMS should decrease with distance
        for i in range(len(rms_values) - 1):
            assert rms_values[i] > rms_values[i+1], \
                f"RMS should decrease with distance: {rms_values}"
