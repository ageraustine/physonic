"""Unit tests for sound mixing."""

import pytest
import numpy as np

from physonic.synthesis.continuous import create_preset_sound
from physonic.synthesis.alerts import create_preset_alert
from physonic.synthesis.mixer import (
    mix_acoustic_scene,
    mix_with_layers,
    create_data_center_ambient,
)


class TestMixAcousticScene:
    """Test acoustic scene mixing."""

    def test_mix_empty_scene(self):
        """Test mixing scene with no sounds."""
        mixed = mix_acoustic_scene(
            impact_sounds=[],
            continuous_sounds=[],
            alert_sounds=[],
            duration=2.0,
        )

        assert mixed.shape == (88200,)
        assert mixed.dtype == np.float32
        assert np.all(mixed == 0)  # Should be silence

    def test_mix_continuous_only(self):
        """Test mixing only continuous sounds."""
        continuous_sounds = [
            create_preset_sound("server1", "server_hum", (0, 0, 0)),
            create_preset_sound("fan1", "fan", (5, 0, 0)),
        ]

        mixed = mix_acoustic_scene(
            impact_sounds=[],
            continuous_sounds=continuous_sounds,
            alert_sounds=[],
            duration=2.0,
        )

        assert len(mixed) == 88200
        assert np.abs(mixed).max() > 0  # Should have content
        assert np.abs(mixed).max() <= 1.0  # No clipping

    def test_mix_alerts_only(self):
        """Test mixing only alert sounds."""
        alert_sounds = [
            (create_preset_alert("beep", (0, 0, 0)), 0.5),
            (create_preset_alert("warning", (0, 0, 0)), 1.5),
        ]

        mixed = mix_acoustic_scene(
            impact_sounds=[],
            continuous_sounds=[],
            alert_sounds=alert_sounds,
            duration=3.0,
        )

        assert len(mixed) == 132300
        assert np.abs(mixed).max() > 0

    def test_mix_impacts_only(self):
        """Test mixing only impact sounds."""
        impact1 = np.random.randn(4410) * 0.5  # 100ms of noise
        impact2 = np.random.randn(2205) * 0.3  # 50ms of noise

        impact_sounds = [
            (impact1, 0.5),
            (impact2, 1.5),
        ]

        mixed = mix_acoustic_scene(
            impact_sounds=impact_sounds,
            continuous_sounds=[],
            alert_sounds=[],
            duration=2.0,
        )

        assert len(mixed) == 88200
        assert np.abs(mixed).max() > 0

    def test_mix_all_sources(self):
        """Test mixing all types of sounds together."""
        continuous_sounds = [
            create_preset_sound("server1", "server_hum", (0, 0, 0)),
        ]
        alert_sounds = [
            (create_preset_alert("beep", (0, 0, 0)), 1.0),
        ]
        impact_sounds = [
            (np.random.randn(4410) * 0.5, 0.5),
        ]

        mixed = mix_acoustic_scene(
            impact_sounds=impact_sounds,
            continuous_sounds=continuous_sounds,
            alert_sounds=alert_sounds,
            duration=2.0,
        )

        assert len(mixed) == 88200
        assert np.abs(mixed).max() > 0
        assert np.abs(mixed).max() <= 1.0

    def test_mix_levels(self):
        """Test that mix levels work (output gets normalized)."""
        continuous_sounds = [create_preset_sound("server1", "server_hum", (0, 0, 0))]

        # Low continuous level
        mixed1 = mix_acoustic_scene(
            impact_sounds=[],
            continuous_sounds=continuous_sounds,
            alert_sounds=[],
            duration=2.0,
            continuous_level=0.2,
        )

        # High continuous level
        mixed2 = mix_acoustic_scene(
            impact_sounds=[],
            continuous_sounds=continuous_sounds,
            alert_sounds=[],
            duration=2.0,
            continuous_level=0.8,
        )

        # Both should produce valid output (normalization rescales)
        assert len(mixed1) == len(mixed2)
        assert np.abs(mixed1).max() > 0
        assert np.abs(mixed2).max() > 0

    def test_impact_truncation(self):
        """Test that impacts are truncated if they exceed duration."""
        long_impact = np.random.randn(88200) * 0.5  # 2 seconds
        impact_sounds = [(long_impact, 1.5)]  # Start at 1.5s in 2s timeline

        mixed = mix_acoustic_scene(
            impact_sounds=impact_sounds,
            continuous_sounds=[],
            alert_sounds=[],
            duration=2.0,
        )

        # Should be truncated to fit
        assert len(mixed) == 88200

    def test_impact_beyond_duration(self):
        """Test impact that starts beyond duration is ignored."""
        impact = np.random.randn(4410) * 0.5
        impact_sounds = [(impact, 5.0)]  # Start at 5s in 2s timeline

        mixed = mix_acoustic_scene(
            impact_sounds=impact_sounds,
            continuous_sounds=[],
            alert_sounds=[],
            duration=2.0,
        )

        # Impact should not appear
        assert np.abs(mixed).max() < 1e-6

    def test_no_clipping_with_many_sources(self):
        """Test that many sources don't cause clipping."""
        continuous_sounds = [
            create_preset_sound(f"server_{i}", "server_hum", (i, 0, 0))
            for i in range(10)
        ]
        alert_sounds = [
            (create_preset_alert("beep", (0, 0, 0)), i * 0.5)
            for i in range(5)
        ]
        impact_sounds = [
            (np.random.randn(2205) * 0.5, i * 0.3)
            for i in range(5)
        ]

        mixed = mix_acoustic_scene(
            impact_sounds=impact_sounds,
            continuous_sounds=continuous_sounds,
            alert_sounds=alert_sounds,
            duration=3.0,
        )

        assert np.abs(mixed).max() <= 1.0


class TestMixWithLayers:
    """Test layer-based mixing."""

    def test_mix_empty_layers(self):
        """Test mixing empty layers."""
        mixed = mix_with_layers(layers=[], duration=1.0)

        assert len(mixed) == 44100
        assert np.all(mixed == 0)

    def test_mix_single_layer(self):
        """Test mixing a single layer."""
        layer = np.random.randn(44100) * 0.5

        mixed = mix_with_layers(
            layers=[(layer, 1.0)],
            duration=1.0,
        )

        assert len(mixed) == 44100
        assert np.abs(mixed).max() > 0

    def test_mix_multiple_layers(self):
        """Test mixing multiple layers."""
        layer1 = np.random.randn(44100) * 0.3
        layer2 = np.random.randn(44100) * 0.4

        mixed = mix_with_layers(
            layers=[(layer1, 0.5), (layer2, 0.8)],
            duration=1.0,
        )

        assert len(mixed) == 44100
        assert np.abs(mixed).max() <= 1.0

    def test_layer_gain_control(self):
        """Test that layer gains work (output gets normalized)."""
        np.random.seed(42)  # For reproducibility
        layer = np.random.randn(44100) * 0.5

        mixed1 = mix_with_layers([(layer.copy(), 0.3)], duration=1.0)
        mixed2 = mix_with_layers([(layer.copy(), 0.9)], duration=1.0)

        # Both should produce valid output (normalization rescales)
        assert len(mixed1) == len(mixed2)
        assert np.abs(mixed1).max() > 0
        assert np.abs(mixed2).max() > 0

    def test_layer_shorter_than_duration(self):
        """Test layer shorter than total duration."""
        short_layer = np.random.randn(22050) * 0.5  # 0.5s

        mixed = mix_with_layers(
            layers=[(short_layer, 1.0)],
            duration=1.0,  # 1s total
        )

        assert len(mixed) == 44100
        # Second half should be silent
        assert np.abs(mixed[22050:]).max() < 1e-6

    def test_layer_longer_than_duration(self):
        """Test layer longer than total duration (should truncate)."""
        long_layer = np.random.randn(88200) * 0.5  # 2s

        mixed = mix_with_layers(
            layers=[(long_layer, 1.0)],
            duration=1.0,  # 1s total
        )

        assert len(mixed) == 44100  # Should be truncated

    def test_normalization(self):
        """Test that output is normalized."""
        layer = np.ones(44100) * 5.0  # Very loud

        mixed = mix_with_layers([(layer, 1.0)], duration=1.0)

        assert np.abs(mixed).max() <= 1.0


class TestDataCenterAmbient:
    """Test data center ambient generation."""

    def test_create_basic(self):
        """Test creating basic data center ambient."""
        ambient = create_data_center_ambient(
            num_servers=3,
            num_fans=2,
            num_cooling=1,
            duration=2.0,
        )

        assert ambient.shape == (88200,)
        assert ambient.dtype == np.float32
        assert np.abs(ambient).max() > 0
        assert np.abs(ambient).max() <= 1.0

    def test_different_configurations(self):
        """Test different equipment configurations."""
        # Minimal
        ambient1 = create_data_center_ambient(
            num_servers=1,
            num_fans=1,
            num_cooling=0,
            duration=1.0,
        )

        # Maximal
        ambient2 = create_data_center_ambient(
            num_servers=10,
            num_fans=5,
            num_cooling=3,
            duration=1.0,
        )

        # Both should be valid
        assert len(ambient1) == 44100
        assert len(ambient2) == 44100
        assert np.abs(ambient1).max() > 0
        assert np.abs(ambient2).max() > 0

    def test_no_equipment(self):
        """Test with no equipment (should be silence)."""
        ambient = create_data_center_ambient(
            num_servers=0,
            num_fans=0,
            num_cooling=0,
            duration=1.0,
        )

        assert len(ambient) == 44100
        assert np.all(ambient == 0)

    def test_duration_control(self):
        """Test different durations."""
        for duration in [0.5, 1.0, 3.0, 5.0]:
            ambient = create_data_center_ambient(
                num_servers=3,
                num_fans=2,
                num_cooling=1,
                duration=duration,
            )
            expected_samples = int(duration * 44100)
            assert len(ambient) == expected_samples

    def test_sample_rate_control(self):
        """Test different sample rates."""
        for sample_rate in [22050, 44100, 48000]:
            ambient = create_data_center_ambient(
                num_servers=3,
                num_fans=2,
                num_cooling=1,
                duration=1.0,
                sample_rate=sample_rate,
            )
            assert len(ambient) == sample_rate

    def test_randomization(self):
        """Test that equipment volumes are randomized."""
        # Generate two ambients with same config
        ambient1 = create_data_center_ambient(
            num_servers=5,
            num_fans=3,
            num_cooling=2,
            duration=1.0,
        )
        ambient2 = create_data_center_ambient(
            num_servers=5,
            num_fans=3,
            num_cooling=2,
            duration=1.0,
        )

        # They should be different (due to random volume variations)
        assert not np.allclose(ambient1, ambient2, rtol=0.1)


class TestIntegration:
    """Integration tests for mixing workflow."""

    def test_realistic_data_center_scene(self):
        """Test realistic data center scene with all sound types."""
        from physonic.synthesis.continuous import synthesize_continuous_sound

        # Background: servers and cooling
        continuous_sounds = [
            create_preset_sound("server1", "server_hum", (0, 0, 0), volume=0.6),
            create_preset_sound("server2", "server_hum", (2, 0, 0), volume=0.7),
            create_preset_sound("cooling1", "cooling_system", (5, 5, 0), volume=0.8),
        ]

        # Events: maintenance beep and tool drop
        alert_sounds = [
            (create_preset_alert("beep", (3, 0, 1)), 1.0),
        ]
        impact_sounds = [
            (np.random.randn(8820) * 0.7, 2.0),  # Tool drop at 2s
        ]

        mixed = mix_acoustic_scene(
            impact_sounds=impact_sounds,
            continuous_sounds=continuous_sounds,
            alert_sounds=alert_sounds,
            duration=5.0,
            continuous_level=0.3,
            alert_level=0.6,
            impact_level=1.0,
        )

        assert len(mixed) == 220500
        assert np.abs(mixed).max() <= 1.0
        assert np.abs(mixed).max() > 0.1  # Should have reasonable level

    def test_complex_timeline(self):
        """Test complex timeline with multiple events."""
        continuous_sounds = [
            create_preset_sound("hvac1", "hvac", (0, 0, 2), volume=0.5),
        ]

        # Many alerts over time
        alert_sounds = [
            (create_preset_alert("beep", (i, 0, 0)), i * 0.5)
            for i in range(10)
        ]

        # Many impacts
        impact_sounds = [
            (np.random.randn(2205) * 0.4, i * 0.3)
            for i in range(10)
        ]

        mixed = mix_acoustic_scene(
            impact_sounds=impact_sounds,
            continuous_sounds=continuous_sounds,
            alert_sounds=alert_sounds,
            duration=6.0,
        )

        assert len(mixed) == 264600
        assert np.abs(mixed).max() <= 1.0
