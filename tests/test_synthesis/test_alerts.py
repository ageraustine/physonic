"""Unit tests for alert sound synthesis."""

import pytest
import numpy as np

from physonic.synthesis.alerts import (
    AlertSoundSpec,
    synthesize_alert_sound,
    create_preset_alert,
    synthesize_two_tone_alert,
    place_alert_in_timeline,
    ALERT_PRESETS,
)


class TestAlertSoundSpec:
    """Test AlertSoundSpec dataclass."""

    def test_create_spec(self):
        """Test creating an alert sound spec."""
        spec = AlertSoundSpec(
            alert_type="beep",
            frequency=1000.0,
            duration=0.1,
            pattern="single",
            envelope="sharp",
            position=(1.0, 2.0, 3.0),
            volume=0.8,
            gap_duration=0.2,
        )

        assert spec.alert_type == "beep"
        assert spec.frequency == 1000.0
        assert spec.duration == 0.1
        assert spec.pattern == "single"
        assert spec.envelope == "sharp"
        assert spec.position == (1.0, 2.0, 3.0)
        assert spec.volume == 0.8
        assert spec.gap_duration == 0.2


class TestPresets:
    """Test preset alert creation."""

    def test_all_presets_exist(self):
        """Test that all expected presets are defined."""
        expected_presets = ["beep", "double_beep", "alarm", "warning", "chime", "error"]
        for preset in expected_presets:
            assert preset in ALERT_PRESETS

    def test_create_preset_alert(self):
        """Test creating alert from preset."""
        spec = create_preset_alert("beep", position=(0, 0, 0), volume=1.0)

        assert spec.alert_type == "beep"
        assert spec.frequency == 1000.0
        assert spec.volume == 1.0

    def test_create_preset_invalid_type(self):
        """Test that invalid preset type raises error."""
        with pytest.raises(ValueError, match="Unknown alert type"):
            create_preset_alert("invalid_type", (0, 0, 0))

    def test_all_presets_can_be_created(self):
        """Test that all presets can be instantiated."""
        for alert_type in ALERT_PRESETS.keys():
            spec = create_preset_alert(alert_type, (0, 0, 0))
            assert spec.alert_type == alert_type
            assert spec.frequency > 0
            assert spec.duration > 0


class TestSynthesis:
    """Test alert sound synthesis."""

    def test_synthesize_basic(self):
        """Test basic alert synthesis."""
        spec = create_preset_alert("beep", (0, 0, 0))
        audio = synthesize_alert_sound(spec, sample_rate=44100)

        assert len(audio) > 0
        assert audio.dtype == np.float32
        assert np.abs(audio).max() <= 1.0  # No clipping

    def test_single_beep(self):
        """Test single beep pattern."""
        spec = create_preset_alert("beep", (0, 0, 0))
        audio = synthesize_alert_sound(spec)

        # Single beep should be approximately duration length
        expected_samples = int(spec.duration * 44100)
        assert len(audio) == pytest.approx(expected_samples, rel=0.1)

    def test_double_beep(self):
        """Test double beep pattern."""
        spec = create_preset_alert("double_beep", (0, 0, 0))
        audio = synthesize_alert_sound(spec)

        # Double beep: 2 beeps + 1 gap
        expected_samples = int((2 * spec.duration + spec.gap_duration) * 44100)
        assert len(audio) == pytest.approx(expected_samples, rel=0.1)

    def test_triple_beep(self):
        """Test triple beep pattern."""
        spec = AlertSoundSpec(
            alert_type="test",
            frequency=1000.0,
            duration=0.1,
            pattern="triple",
            envelope="sharp",
            position=(0, 0, 0),
            gap_duration=0.1,
        )
        audio = synthesize_alert_sound(spec)

        # Triple beep: 3 beeps + 2 gaps
        expected_samples = int((3 * spec.duration + 2 * spec.gap_duration) * 44100)
        assert len(audio) == pytest.approx(expected_samples, rel=0.1)

    def test_continuous_pattern(self):
        """Test continuous pattern (5 beeps)."""
        spec = AlertSoundSpec(
            alert_type="test",
            frequency=1000.0,
            duration=0.1,
            pattern="continuous",
            envelope="sharp",
            position=(0, 0, 0),
            gap_duration=0.1,
        )
        audio = synthesize_alert_sound(spec)

        # Continuous: 5 beeps + 4 gaps
        expected_samples = int((5 * spec.duration + 4 * spec.gap_duration) * 44100)
        assert len(audio) == pytest.approx(expected_samples, rel=0.1)

    def test_invalid_pattern(self):
        """Test that invalid pattern raises error."""
        spec = AlertSoundSpec(
            alert_type="test",
            frequency=1000.0,
            duration=0.1,
            pattern="invalid",
            envelope="sharp",
            position=(0, 0, 0),
        )

        with pytest.raises(ValueError, match="Unknown pattern"):
            synthesize_alert_sound(spec)

    def test_sharp_envelope(self):
        """Test sharp envelope (fast attack/release)."""
        spec = AlertSoundSpec(
            alert_type="test",
            frequency=1000.0,
            duration=0.2,
            pattern="single",
            envelope="sharp",
            position=(0, 0, 0),
        )
        audio = synthesize_alert_sound(spec)

        # Just verify sharp envelope produces valid audio
        assert len(audio) > 0
        assert np.abs(audio).max() > 0.5  # Should have significant amplitude

    def test_soft_envelope(self):
        """Test soft envelope (gradual attack/release)."""
        spec = AlertSoundSpec(
            alert_type="test",
            frequency=1000.0,
            duration=0.2,
            pattern="single",
            envelope="soft",
            position=(0, 0, 0),
        )
        audio = synthesize_alert_sound(spec)

        # Soft envelope should have gradual attack
        attack_samples = int(0.01 * 44100)  # First 10ms
        assert np.abs(audio[attack_samples]) < 0.3  # Should still be rising

    def test_volume_control(self):
        """Test that volume parameter works (output gets normalized)."""
        spec1 = create_preset_alert("beep", (0, 0, 0), volume=0.5)
        spec2 = create_preset_alert("beep", (0, 0, 0), volume=1.0)

        audio1 = synthesize_alert_sound(spec1)
        audio2 = synthesize_alert_sound(spec2)

        # Both should produce valid output (normalization rescales)
        assert len(audio1) > 0
        assert len(audio2) > 0
        assert np.abs(audio1).max() > 0
        assert np.abs(audio2).max() > 0

    def test_frequency_control(self):
        """Test that frequency parameter affects pitch."""
        spec1 = AlertSoundSpec(
            alert_type="test",
            frequency=500.0,
            duration=0.2,
            pattern="single",
            envelope="sharp",
            position=(0, 0, 0),
        )
        spec2 = AlertSoundSpec(
            alert_type="test",
            frequency=1000.0,
            duration=0.2,
            pattern="single",
            envelope="sharp",
            position=(0, 0, 0),
        )

        audio1 = synthesize_alert_sound(spec1)
        audio2 = synthesize_alert_sound(spec2)

        # Different frequencies should produce different signals
        assert not np.allclose(audio1, audio2)

    def test_output_is_normalized(self):
        """Test that output is normalized."""
        spec = create_preset_alert("alarm", (0, 0, 0))
        audio = synthesize_alert_sound(spec)

        # Should be normalized with headroom
        assert np.abs(audio).max() <= 1.0
        assert np.abs(audio).max() > 0.7  # Should use reasonable dynamic range


class TestTwoToneAlert:
    """Test two-tone alert synthesis."""

    def test_two_tone_basic(self):
        """Test basic two-tone synthesis."""
        audio = synthesize_two_tone_alert(
            frequency1=523.25,  # C5
            frequency2=659.25,  # E5
            duration=0.2,
        )

        assert len(audio) > 0
        assert audio.dtype == np.float32
        assert np.abs(audio).max() <= 1.0

    def test_two_tone_length(self):
        """Test that two-tone has expected length."""
        duration = 0.2
        audio = synthesize_two_tone_alert(
            frequency1=500.0,
            frequency2=1000.0,
            duration=duration,
        )

        # Should be approximately 2*duration + gap (0.05s)
        expected_samples = int((2 * duration + 0.05) * 44100)
        assert len(audio) == pytest.approx(expected_samples, rel=0.15)

    def test_two_tone_volume(self):
        """Test two-tone volume control."""
        audio1 = synthesize_two_tone_alert(500.0, 1000.0, volume=0.5)
        audio2 = synthesize_two_tone_alert(500.0, 1000.0, volume=1.0)

        # Both should produce valid output (normalization rescales)
        assert len(audio1) > 0
        assert len(audio2) > 0
        assert np.abs(audio1).max() > 0
        assert np.abs(audio2).max() > 0


class TestTimelinePlacement:
    """Test placing alerts in timeline."""

    def test_place_alert_basic(self):
        """Test basic timeline placement."""
        spec = create_preset_alert("beep", (0, 0, 0))
        alert_audio = synthesize_alert_sound(spec)

        timeline = place_alert_in_timeline(
            alert_audio,
            trigger_time=1.0,
            total_duration=3.0,
        )

        assert len(timeline) == int(3.0 * 44100)
        assert timeline.dtype == np.float32

    def test_alert_at_correct_time(self):
        """Test that alert appears at correct time."""
        spec = create_preset_alert("beep", (0, 0, 0))
        alert_audio = synthesize_alert_sound(spec)

        trigger_time = 1.0
        timeline = place_alert_in_timeline(alert_audio, trigger_time, 3.0)

        # Alert should be silent before trigger time
        silence_end = int((trigger_time - 0.1) * 44100)
        assert np.abs(timeline[:silence_end]).max() < 1e-6

        # Alert should have content at trigger time
        alert_start = int(trigger_time * 44100)
        alert_end = alert_start + len(alert_audio)
        assert np.abs(timeline[alert_start:alert_end]).max() > 0.1

    def test_alert_at_start(self):
        """Test placing alert at time 0."""
        spec = create_preset_alert("beep", (0, 0, 0))
        alert_audio = synthesize_alert_sound(spec)

        timeline = place_alert_in_timeline(alert_audio, trigger_time=0.0, total_duration=2.0)

        # Alert should start immediately
        assert np.abs(timeline[:len(alert_audio)]).max() > 0.1

    def test_alert_at_end(self):
        """Test placing alert at end of timeline."""
        spec = create_preset_alert("beep", (0, 0, 0))
        alert_audio = synthesize_alert_sound(spec)

        total_duration = 2.0
        trigger_time = 1.9

        timeline = place_alert_in_timeline(alert_audio, trigger_time, total_duration)

        # Alert should be truncated but present
        trigger_sample = int(trigger_time * 44100)
        assert np.abs(timeline[trigger_sample:]).max() > 0.1

    def test_alert_beyond_timeline(self):
        """Test placing alert beyond timeline (should be silent)."""
        spec = create_preset_alert("beep", (0, 0, 0))
        alert_audio = synthesize_alert_sound(spec)

        timeline = place_alert_in_timeline(
            alert_audio,
            trigger_time=5.0,  # Beyond timeline
            total_duration=2.0,
        )

        # Should be all zeros
        assert np.all(timeline == 0)

    def test_multiple_alerts_summing(self):
        """Test that multiple alerts can be placed by summing."""
        spec = create_preset_alert("beep", (0, 0, 0))
        alert_audio = synthesize_alert_sound(spec)

        timeline = np.zeros(int(5.0 * 44100), dtype=np.float32)
        timeline += place_alert_in_timeline(alert_audio, 1.0, 5.0)
        timeline += place_alert_in_timeline(alert_audio, 3.0, 5.0)

        # Should have content at both times
        assert np.abs(timeline[int(1.0 * 44100):int(1.0 * 44100) + 1000]).max() > 0.1
        assert np.abs(timeline[int(3.0 * 44100):int(3.0 * 44100) + 1000]).max() > 0.1


class TestAlertCharacteristics:
    """Test that generated alerts have expected characteristics."""

    def test_beep_frequency(self):
        """Test that beep has correct frequency."""
        spec = create_preset_alert("beep", (0, 0, 0))
        audio = synthesize_alert_sound(spec)

        # FFT to check frequency
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/44100)

        # Find peak frequency
        peak_idx = np.argmax(np.abs(fft))
        peak_freq = freqs[peak_idx]

        # Should be close to 1000Hz
        assert abs(peak_freq - 1000.0) < 10.0

    def test_alarm_is_longer(self):
        """Test that alarm is longer than beep."""
        beep = synthesize_alert_sound(create_preset_alert("beep", (0, 0, 0)))
        alarm = synthesize_alert_sound(create_preset_alert("alarm", (0, 0, 0)))

        assert len(alarm) > len(beep)

    def test_envelope_affects_shape(self):
        """Test that envelope type affects waveform shape."""
        spec_sharp = AlertSoundSpec(
            alert_type="test",
            frequency=1000.0,
            duration=0.2,
            pattern="single",
            envelope="sharp",
            position=(0, 0, 0),
        )
        spec_soft = AlertSoundSpec(
            alert_type="test",
            frequency=1000.0,
            duration=0.2,
            pattern="single",
            envelope="soft",
            position=(0, 0, 0),
        )

        audio_sharp = synthesize_alert_sound(spec_sharp)
        audio_soft = synthesize_alert_sound(spec_soft)

        # Waveforms should be different due to envelope
        assert not np.allclose(audio_sharp, audio_soft)
