"""Unit tests for physonic.spatial.hrtf module."""
import numpy as np
import pytest
from physonic.spatial.hrtf import HRTFSet, load_kemar_hrtf, interpolate_hrtf, convolve_hrtf


class TestLoadKemarHRTF:
    """Tests for load_kemar_hrtf function."""

    def test_load_hrtf_mini(self, hrtf_mini_path):
        """Test loading minimal HRTF dataset."""
        hrtf = load_kemar_hrtf(str(hrtf_mini_path))

        assert isinstance(hrtf, HRTFSet)
        assert hrtf.sample_rate == 44100
        assert hrtf.hrir_length == 128
        assert len(hrtf.elevations) == 1
        assert 0 in hrtf.elevations

    def test_hrtf_contains_expected_measurements(self, hrtf_mini):
        """Test that HRTF contains expected angle measurements."""
        assert (0, 0) in hrtf_mini.hrir_left
        assert (0, 45) in hrtf_mini.hrir_left
        assert (0, 90) in hrtf_mini.hrir_left
        assert (0, 135) in hrtf_mini.hrir_left
        assert (0, 180) in hrtf_mini.hrir_left

        # Right ear should have same keys
        assert (0, 0) in hrtf_mini.hrir_right
        assert (0, 90) in hrtf_mini.hrir_right

    def test_hrir_shape_and_type(self, hrtf_mini):
        """Test HRIR arrays have correct shape and dtype."""
        hrir_l = hrtf_mini.hrir_left[(0, 0)]
        hrir_r = hrtf_mini.hrir_right[(0, 0)]

        assert hrir_l.shape == (128,)
        assert hrir_r.shape == (128,)
        assert hrir_l.dtype == np.float32
        assert hrir_r.dtype == np.float32

    def test_hrir_values_finite(self, hrtf_mini):
        """Test that all HRIR values are finite."""
        for key in hrtf_mini.hrir_left:
            assert np.isfinite(hrtf_mini.hrir_left[key]).all()
            assert np.isfinite(hrtf_mini.hrir_right[key]).all()

    def test_load_nonexistent_path_raises(self):
        """Test that loading nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="HRTF dataset path not found"):
            load_kemar_hrtf("/nonexistent/path")

    def test_azimuths_dict_structure(self, hrtf_mini):
        """Test that azimuths dictionary is correctly structured."""
        assert 0 in hrtf_mini.azimuths
        azims = hrtf_mini.azimuths[0]
        assert isinstance(azims, np.ndarray)
        assert len(azims) == 5
        np.testing.assert_array_equal(azims, [0, 45, 90, 135, 180])


class TestInterpolateHRTF:
    """Tests for interpolate_hrtf function."""

    def test_interpolate_exact_angle(self, hrtf_mini):
        """Test interpolation at exact measured angle returns exact HRIR."""
        hrir_l, hrir_r = interpolate_hrtf(hrtf_mini, azimuth=0, elevation=0)

        np.testing.assert_array_equal(hrir_l, hrtf_mini.hrir_left[(0, 0)])
        np.testing.assert_array_equal(hrir_r, hrtf_mini.hrir_right[(0, 0)])

    def test_interpolate_nearest_neighbor(self, hrtf_mini):
        """Test nearest-neighbor interpolation for intermediate angles."""
        # 47° should snap to 45°
        hrir_l_47, hrir_r_47 = interpolate_hrtf(hrtf_mini, azimuth=47, elevation=0)
        hrir_l_45, hrir_r_45 = interpolate_hrtf(hrtf_mini, azimuth=45, elevation=0)

        np.testing.assert_array_equal(hrir_l_47, hrir_l_45)
        np.testing.assert_array_equal(hrir_r_47, hrir_r_45)

    def test_interpolate_snaps_to_nearest(self, hrtf_mini):
        """Test that interpolation snaps to nearest measured angle."""
        # 22° is closer to 0° than to 45°
        hrir_l_22, _ = interpolate_hrtf(hrtf_mini, azimuth=22, elevation=0)
        hrir_l_0, _ = interpolate_hrtf(hrtf_mini, azimuth=0, elevation=0)

        np.testing.assert_array_equal(hrir_l_22, hrir_l_0)

        # 23° is closer to 45° than to 0°
        hrir_l_23, _ = interpolate_hrtf(hrtf_mini, azimuth=23, elevation=0)
        hrir_l_45, _ = interpolate_hrtf(hrtf_mini, azimuth=45, elevation=0)

        np.testing.assert_array_equal(hrir_l_23, hrir_l_45)

    def test_interpolate_returns_correct_shape(self, hrtf_mini):
        """Test that interpolate_hrtf returns correct shapes."""
        hrir_l, hrir_r = interpolate_hrtf(hrtf_mini, azimuth=90, elevation=0)

        assert hrir_l.shape == (128,)
        assert hrir_r.shape == (128,)

    def test_interpolate_negative_azimuth(self, hrtf_mini):
        """Test interpolation with negative azimuth angles."""
        # -45° should snap to 0° (nearest in our mini dataset)
        hrir_l, _ = interpolate_hrtf(hrtf_mini, azimuth=-45, elevation=0)
        hrir_l_0, _ = interpolate_hrtf(hrtf_mini, azimuth=0, elevation=0)

        np.testing.assert_array_equal(hrir_l, hrir_l_0)


class TestConvolveHRTF:
    """Tests for convolve_hrtf function."""

    def test_convolve_output_shape(self, hrtf_mini):
        """Test that convolution output has correct shape."""
        mono = np.random.randn(1000)
        hrir_l, hrir_r = interpolate_hrtf(hrtf_mini, azimuth=0, elevation=0)

        stereo = convolve_hrtf(mono, hrir_l, hrir_r)

        # Output should be (N + M - 1, 2)
        expected_len = len(mono) + len(hrir_l) - 1
        assert stereo.shape == (expected_len, 2)

    def test_convolve_output_dtype(self, hrtf_mini):
        """Test that convolution output has correct dtype."""
        mono = np.random.randn(1000)
        hrir_l, hrir_r = interpolate_hrtf(hrtf_mini, azimuth=0, elevation=0)

        stereo = convolve_hrtf(mono, hrir_l, hrir_r)

        assert stereo.dtype == np.float32

    def test_convolve_stereo_channels(self, hrtf_mini):
        """Test that convolution produces distinct left/right channels."""
        mono = np.random.randn(1000)
        hrir_l, hrir_r = interpolate_hrtf(hrtf_mini, azimuth=90, elevation=0)

        stereo = convolve_hrtf(mono, hrir_l, hrir_r)

        # Channels should be different (unless HRIRs are identical)
        assert not np.array_equal(stereo[:, 0], stereo[:, 1])

    def test_convolve_itd_detection(self, hrtf_mini, impulse_1sec):
        """Test that ITD can be detected in convolved output."""
        # Use 90° azimuth (left) which should have positive ITD
        hrir_l, hrir_r = interpolate_hrtf(hrtf_mini, azimuth=90, elevation=0)
        stereo = convolve_hrtf(impulse_1sec, hrir_l, hrir_r)

        # Find peaks in first 500 samples (avoid late reflections)
        peak_l = np.argmax(np.abs(stereo[:500, 0]))
        peak_r = np.argmax(np.abs(stereo[:500, 1]))

        itd_samples = peak_l - peak_r
        itd_ms = itd_samples / 44.1  # Convert to milliseconds

        # For 90° azimuth, ITD should be positive (left delayed)
        # and roughly 0.5-0.8 ms based on our synthetic HRTF
        assert itd_ms > 0, f"ITD should be positive for 90° azimuth, got {itd_ms:.2f} ms"
        assert 0.3 < itd_ms < 1.0, f"ITD {itd_ms:.2f} ms outside expected range [0.3, 1.0] ms"

    def test_convolve_with_zero_input(self, hrtf_mini):
        """Test convolution with zero input."""
        mono = np.zeros(1000)
        hrir_l, hrir_r = interpolate_hrtf(hrtf_mini, azimuth=0, elevation=0)

        stereo = convolve_hrtf(mono, hrir_l, hrir_r)

        # Output should be all zeros
        np.testing.assert_allclose(stereo, 0.0, atol=1e-10)

    def test_convolve_energy_conservation(self, hrtf_mini):
        """Test that convolution roughly conserves energy."""
        mono = np.random.randn(1000)
        hrir_l, hrir_r = interpolate_hrtf(hrtf_mini, azimuth=0, elevation=0)

        stereo = convolve_hrtf(mono, hrir_l, hrir_r)

        # Energy in mono signal
        energy_mono = np.sum(mono ** 2)

        # Energy in stereo (sum of both channels)
        energy_stereo = np.sum(stereo[:, 0] ** 2) + np.sum(stereo[:, 1] ** 2)

        # HRTF energy
        energy_hrtf = np.sum(hrir_l ** 2) + np.sum(hrir_r ** 2)

        # Convolution energy should be roughly mono_energy * hrtf_energy
        expected_energy = energy_mono * energy_hrtf

        # Allow 50% tolerance due to numerical precision and HRTF characteristics
        assert 0.5 * expected_energy < energy_stereo < 2.0 * expected_energy


@pytest.mark.integration
class TestHRTFIntegration:
    """Integration tests for HRTF module."""

    def test_load_interpolate_convolve_pipeline(self, hrtf_mini_path, impulse_1sec):
        """Test full pipeline: load → interpolate → convolve."""
        # Load
        hrtf = load_kemar_hrtf(str(hrtf_mini_path))

        # Interpolate
        hrir_l, hrir_r = interpolate_hrtf(hrtf, azimuth=45, elevation=0)

        # Convolve
        stereo = convolve_hrtf(impulse_1sec, hrir_l, hrir_r)

        # Validate output
        assert stereo.shape[1] == 2
        assert np.isfinite(stereo).all()
        assert np.abs(stereo).max() > 0  # Not silent
