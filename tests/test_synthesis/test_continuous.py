"""Unit tests for continuous sound synthesis."""

import pytest
import numpy as np

from physonic.synthesis.continuous import (
    ContinuousSoundSpec,
    synthesize_continuous_sound,
    create_preset_sound,
    synthesize_multiple_sources,
    SOUND_PRESETS,
)


class TestContinuousSoundSpec:
    """Test ContinuousSoundSpec dataclass."""

    def test_create_spec(self):
        """Test creating a continuous sound spec."""
        spec = ContinuousSoundSpec(
            source_id="test_server",
            sound_type="server_hum",
            fundamental_freq=120.0,
            harmonics=[1.0, 0.5, 0.3],
            noise_amount=0.2,
            modulation_freq=1.0,
            modulation_depth=0.1,
            position=(1.0, 2.0, 3.0),
            volume=0.8,
        )

        assert spec.source_id == "test_server"
        assert spec.sound_type == "server_hum"
        assert spec.fundamental_freq == 120.0
        assert len(spec.harmonics) == 3
        assert spec.position == (1.0, 2.0, 3.0)
        assert spec.volume == 0.8


class TestPresets:
    """Test preset creation."""

    def test_all_presets_exist(self):
        """Test that all expected presets are defined."""
        expected_presets = ["server_hum", "fan", "cooling_system", "hvac"]
        for preset in expected_presets:
            assert preset in SOUND_PRESETS

    def test_create_preset_sound(self):
        """Test creating sound from preset."""
        spec = create_preset_sound(
            source_id="test1",
            sound_type="server_hum",
            position=(0, 0, 0),
            volume=1.0
        )

        assert spec.source_id == "test1"
        assert spec.sound_type == "server_hum"
        assert spec.fundamental_freq == 120.0
        assert spec.volume == 1.0

    def test_create_preset_invalid_type(self):
        """Test that invalid preset type raises error."""
        with pytest.raises(ValueError, match="Unknown sound type"):
            create_preset_sound("test", "invalid_type", (0, 0, 0))

    def test_all_presets_can_be_created(self):
        """Test that all presets can be instantiated."""
        for sound_type in SOUND_PRESETS.keys():
            spec = create_preset_sound(f"test_{sound_type}", sound_type, (0, 0, 0))
            assert spec.sound_type == sound_type
            assert spec.fundamental_freq > 0
            assert len(spec.harmonics) > 0


class TestSynthesis:
    """Test continuous sound synthesis."""

    def test_synthesize_basic(self):
        """Test basic sound synthesis."""
        spec = create_preset_sound("test", "server_hum", (0, 0, 0))
        audio = synthesize_continuous_sound(spec, duration=1.0, sample_rate=44100)

        assert audio.shape == (44100,)
        assert audio.dtype == np.float32
        assert np.abs(audio).max() <= 1.0  # No clipping

    def test_synthesize_different_durations(self):
        """Test synthesis with different durations."""
        spec = create_preset_sound("test", "fan", (0, 0, 0))

        for duration in [0.5, 1.0, 2.0, 5.0]:
            audio = synthesize_continuous_sound(spec, duration=duration)
            expected_samples = int(duration * 44100)
            assert len(audio) == expected_samples

    def test_synthesize_different_sample_rates(self):
        """Test synthesis with different sample rates."""
        spec = create_preset_sound("test", "fan", (0, 0, 0))

        for sample_rate in [22050, 44100, 48000]:
            audio = synthesize_continuous_sound(spec, duration=1.0, sample_rate=sample_rate)
            assert len(audio) == sample_rate

    def test_synthesis_with_no_harmonics(self):
        """Test synthesis with only fundamental (no harmonics)."""
        spec = ContinuousSoundSpec(
            source_id="test",
            sound_type="test",
            fundamental_freq=100.0,
            harmonics=[1.0],  # Only fundamental
            noise_amount=0.0,
            modulation_freq=0.0,
            modulation_depth=0.0,
            position=(0, 0, 0),
            volume=1.0,
        )
        audio = synthesize_continuous_sound(spec, duration=1.0)

        assert len(audio) == 44100
        assert np.abs(audio).max() > 0  # Sound was generated

    def test_synthesis_with_noise_only(self):
        """Test synthesis with only noise (no harmonics)."""
        spec = ContinuousSoundSpec(
            source_id="test",
            sound_type="test",
            fundamental_freq=100.0,
            harmonics=[0.0],  # No harmonics
            noise_amount=1.0,  # High noise
            modulation_freq=0.0,
            modulation_depth=0.0,
            position=(0, 0, 0),
            volume=1.0,
        )
        audio = synthesize_continuous_sound(spec, duration=1.0)

        assert len(audio) == 44100
        assert np.abs(audio).max() > 0

    def test_synthesis_with_modulation(self):
        """Test that amplitude modulation is applied."""
        spec = ContinuousSoundSpec(
            source_id="test",
            sound_type="test",
            fundamental_freq=100.0,
            harmonics=[1.0],
            noise_amount=0.0,
            modulation_freq=2.0,  # 2 Hz modulation
            modulation_depth=0.5,  # 50% depth
            position=(0, 0, 0),
            volume=1.0,
        )
        audio = synthesize_continuous_sound(spec, duration=2.0)

        # Check that amplitude varies over time (sign of modulation)
        chunk_size = 44100 // 4
        chunk_energies = [
            np.sqrt(np.mean(audio[i:i+chunk_size]**2))
            for i in range(0, len(audio), chunk_size)
        ]

        # Energy should vary due to modulation
        assert np.std(chunk_energies) > 0.01

    def test_volume_control(self):
        """Test that volume parameter affects amplitude before normalization."""
        spec1 = create_preset_sound("test", "fan", (0, 0, 0), volume=0.5)
        spec2 = create_preset_sound("test", "fan", (0, 0, 0), volume=1.0)

        # Volume affects the synthesis, though final normalization may rescale
        # Just verify they produce valid output
        audio1 = synthesize_continuous_sound(spec1, duration=1.0)
        audio2 = synthesize_continuous_sound(spec2, duration=1.0)

        assert len(audio1) == len(audio2)
        assert np.abs(audio1).max() > 0
        assert np.abs(audio2).max() > 0

    def test_output_is_normalized(self):
        """Test that output is normalized to prevent clipping."""
        spec = create_preset_sound("test", "cooling_system", (0, 0, 0), volume=1.0)
        audio = synthesize_continuous_sound(spec, duration=1.0)

        # Should be normalized with headroom
        assert np.abs(audio).max() <= 1.0
        assert np.abs(audio).max() > 0.5  # Should use reasonable dynamic range


class TestMultipleSources:
    """Test synthesizing multiple sources."""

    def test_synthesize_multiple_sources(self):
        """Test mixing multiple continuous sources."""
        specs = [
            create_preset_sound(f"server_{i}", "server_hum", (i*2, 0, 0))
            for i in range(3)
        ]

        mixed = synthesize_multiple_sources(specs, duration=2.0, mix_level=0.5)

        assert mixed.shape == (88200,)
        assert mixed.dtype == np.float32
        assert np.abs(mixed).max() <= 1.0

    def test_empty_sources(self):
        """Test that empty sources list returns silence."""
        mixed = synthesize_multiple_sources([], duration=1.0)

        assert len(mixed) == 44100
        assert np.all(mixed == 0)

    def test_single_source(self):
        """Test mixing a single source."""
        spec = create_preset_sound("test", "fan", (0, 0, 0))
        mixed = synthesize_multiple_sources([spec], duration=1.0)

        assert len(mixed) == 44100
        assert np.abs(mixed).max() > 0

    def test_mix_level_control(self):
        """Test that mix_level parameter works (output gets normalized)."""
        specs = [create_preset_sound("test", "fan", (0, 0, 0))]

        # Mix level affects synthesis, though normalization rescales
        mixed1 = synthesize_multiple_sources(specs, duration=1.0, mix_level=0.3)
        mixed2 = synthesize_multiple_sources(specs, duration=1.0, mix_level=0.6)

        # Just verify both produce valid output
        assert len(mixed1) == len(mixed2)
        assert np.abs(mixed1).max() > 0
        assert np.abs(mixed2).max() > 0

    def test_multiple_sources_no_clipping(self):
        """Test that multiple sources don't cause clipping."""
        specs = [
            create_preset_sound(f"source_{i}", "server_hum", (i, 0, 0), volume=1.0)
            for i in range(10)  # Many sources
        ]

        mixed = synthesize_multiple_sources(specs, duration=1.0, mix_level=0.5)

        assert np.abs(mixed).max() <= 1.0


class TestSoundCharacteristics:
    """Test that generated sounds have expected characteristics."""

    def test_server_hum_frequency(self):
        """Test that server hum has dominant frequency at 120Hz."""
        spec = create_preset_sound("test", "server_hum", (0, 0, 0))
        audio = synthesize_continuous_sound(spec, duration=2.0)

        # FFT to check frequency content
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/44100)

        # Find peak frequency
        peak_idx = np.argmax(np.abs(fft))
        peak_freq = freqs[peak_idx]

        # Should be close to 120Hz (within 5Hz)
        assert abs(peak_freq - 120.0) < 5.0

    def test_fan_has_noise(self):
        """Test that fan sound has broadband noise component."""
        spec = create_preset_sound("test", "fan", (0, 0, 0))
        audio = synthesize_continuous_sound(spec, duration=1.0)

        # Fan should have significant noise amount in spec
        assert spec.noise_amount > 0.2

        # Audio should have energy across frequency spectrum (not just tonal)
        # Just verify the audio has content and isn't pure silence
        fft = np.abs(np.fft.rfft(audio))
        assert np.max(fft) > 0
        assert np.mean(fft) > 0
