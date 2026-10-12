"""Unit tests for physonic.spatial.microphone module."""
import numpy as np
import pytest
from physonic.spatial.microphone import BinauralMicrophone


class TestBinauralMicrophone:
    """Tests for BinauralMicrophone class."""

    def test_initialization(self):
        """Test microphone initialization."""
        mic = BinauralMicrophone(
            position=np.array([1.0, 2.0, 3.0]),
            orientation=45.0
        )

        np.testing.assert_array_equal(mic.position, [1.0, 2.0, 3.0])
        assert mic.orientation == 45.0

    def test_compute_angle_front(self, binaural_mic):
        """Test angle computation for source directly in front."""
        source_pos = np.array([1.0, 0.0, 1.0])
        azim, elev = binaural_mic.compute_source_angle(source_pos)

        assert abs(azim - 0.0) < 1.0  # Front (0°)
        assert abs(elev - 0.0) < 1.0  # Horizontal (0°)

    def test_compute_angle_left(self, binaural_mic):
        """Test angle computation for source to the left."""
        source_pos = np.array([0.0, 1.0, 1.0])
        azim, elev = binaural_mic.compute_source_angle(source_pos)

        assert abs(azim - 90.0) < 1.0  # Left (90°)
        assert abs(elev - 0.0) < 1.0   # Horizontal

    def test_compute_angle_right(self, binaural_mic):
        """Test angle computation for source to the right."""
        source_pos = np.array([0.0, -1.0, 1.0])
        azim, elev = binaural_mic.compute_source_angle(source_pos)

        assert abs(azim - (-90.0)) < 1.0  # Right (-90°)
        assert abs(elev - 0.0) < 1.0      # Horizontal

    def test_compute_angle_back(self, binaural_mic):
        """Test angle computation for source behind."""
        source_pos = np.array([-1.0, 0.0, 1.0])
        azim, elev = binaural_mic.compute_source_angle(source_pos)

        # Back can be ±180°
        assert abs(abs(azim) - 180.0) < 1.0
        assert abs(elev - 0.0) < 1.0

    def test_compute_angle_above(self, binaural_mic):
        """Test angle computation for source above."""
        source_pos = np.array([1.0, 0.0, 2.0])
        azim, elev = binaural_mic.compute_source_angle(source_pos)

        assert abs(azim - 0.0) < 1.0   # Front
        assert 40.0 < elev < 50.0      # ~45° above

    def test_compute_angle_below(self, binaural_mic):
        """Test angle computation for source below."""
        source_pos = np.array([1.0, 0.0, 0.0])
        azim, elev = binaural_mic.compute_source_angle(source_pos)

        assert abs(azim - 0.0) < 1.0   # Front
        assert -50.0 < elev < -40.0    # ~-45° below

    def test_compute_angle_front_left(self, binaural_mic):
        """Test angle computation for source at front-left (45°)."""
        source_pos = np.array([1.0, 1.0, 1.0])
        azim, elev = binaural_mic.compute_source_angle(source_pos)

        assert 40.0 < azim < 50.0      # ~45° left
        assert abs(elev - 0.0) < 1.0   # Horizontal

    def test_compute_angle_with_orientation(self):
        """Test angle computation with microphone oriented 90° (facing +Y)."""
        mic = BinauralMicrophone(
            position=np.array([0.0, 0.0, 1.0]),
            orientation=90.0  # Facing +Y (left)
        )

        # Source at world +X should appear to the right in mic frame
        source_pos = np.array([1.0, 0.0, 1.0])
        azim, elev = mic.compute_source_angle(source_pos)

        assert abs(azim - (-90.0)) < 1.0  # Right in mic frame

    def test_compute_angle_orientation_180(self):
        """Test with microphone facing backward (180°)."""
        mic = BinauralMicrophone(
            position=np.array([0.0, 0.0, 1.0]),
            orientation=180.0
        )

        # Source at world +X should appear behind in mic frame
        source_pos = np.array([1.0, 0.0, 1.0])
        azim, elev = mic.compute_source_angle(source_pos)

        # Should be ±180°
        assert abs(abs(azim) - 180.0) < 1.0

    def test_compute_angle_same_position(self, binaural_mic):
        """Test angle computation when source is at mic position."""
        source_pos = binaural_mic.position.copy()
        azim, elev = binaural_mic.compute_source_angle(source_pos)

        # Angle is undefined at zero distance, but should not crash
        # Typically returns 0, 0 or nan depending on implementation
        assert isinstance(azim, float)
        assert isinstance(elev, float)

    def test_compute_angle_return_types(self, binaural_mic):
        """Test that compute_source_angle returns floats."""
        source_pos = np.array([1.0, 1.0, 1.0])
        azim, elev = binaural_mic.compute_source_angle(source_pos)

        assert isinstance(azim, float)
        assert isinstance(elev, float)

    def test_compute_angle_multiple_sources(self, binaural_mic):
        """Test computing angles for multiple source positions."""
        sources = [
            np.array([1.0, 0.0, 1.0]),   # Front
            np.array([0.0, 1.0, 1.0]),   # Left
            np.array([-1.0, 0.0, 1.0]),  # Back
            np.array([0.0, -1.0, 1.0]),  # Right
        ]

        expected_azims = [0.0, 90.0, 180.0, -90.0]

        for source, expected_azim in zip(sources, expected_azims):
            azim, elev = binaural_mic.compute_source_angle(source)

            # Allow ±180° equivalence for back position
            if abs(expected_azim) == 180.0:
                assert abs(abs(azim) - 180.0) < 1.0
            else:
                assert abs(azim - expected_azim) < 1.0

            assert abs(elev - 0.0) < 1.0  # All horizontal


@pytest.mark.integration
class TestMicrophoneIntegration:
    """Integration tests for microphone module."""

    def test_mic_angle_computation_consistency(self):
        """Test that angle computation is consistent across different mic positions."""
        # Create two mics at different positions
        mic1 = BinauralMicrophone(np.array([0.0, 0.0, 1.0]), 0.0)
        mic2 = BinauralMicrophone(np.array([1.0, 0.0, 1.0]), 0.0)

        # Same source position
        source = np.array([2.0, 1.0, 1.0])

        azim1, elev1 = mic1.compute_source_angle(source)
        azim2, elev2 = mic2.compute_source_angle(source)

        # Different positions should give different angles
        assert azim1 != azim2 or elev1 != elev2

    def test_orientation_affects_azimuth_only(self):
        """Test that mic orientation affects azimuth but not elevation."""
        mic_0 = BinauralMicrophone(np.array([0.0, 0.0, 1.0]), 0.0)
        mic_90 = BinauralMicrophone(np.array([0.0, 0.0, 1.0]), 90.0)

        source = np.array([1.0, 0.0, 2.0])  # Front and above

        azim_0, elev_0 = mic_0.compute_source_angle(source)
        azim_90, elev_90 = mic_90.compute_source_angle(source)

        # Azimuth should differ
        assert abs(azim_0 - azim_90) > 10.0

        # Elevation should be same (orientation is yaw only)
        assert abs(elev_0 - elev_90) < 1.0
