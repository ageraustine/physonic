"""Tests for episode loading."""

import pytest
import numpy as np
from pathlib import Path

from physonic.export.episode import load_episode, events_in_frame


def test_events_in_frame():
    """Test filtering events by frame time."""
    events = [
        {"t": 0.1, "impulse": 5.0},
        {"t": 0.15, "impulse": 3.0},
        {"t": 0.5, "impulse": 7.0},
        {"t": 1.0, "impulse": 2.0},
    ]

    # Frame at t=0.0, dt=0.2 should include first two events
    frame_events = events_in_frame(events, frame_t=0.0, dt=0.2)
    assert len(frame_events) == 2
    assert frame_events[0]["t"] == 0.1
    assert frame_events[1]["t"] == 0.15

    # Frame at t=0.5, dt=0.5 should include third event
    frame_events = events_in_frame(events, frame_t=0.5, dt=0.5)
    assert len(frame_events) == 1
    assert frame_events[0]["t"] == 0.5

    # Frame at t=1.0, dt=0.1 should include fourth event
    frame_events = events_in_frame(events, frame_t=1.0, dt=0.1)
    assert len(frame_events) == 1
    assert frame_events[0]["t"] == 1.0

    # Frame with no events
    frame_events = events_in_frame(events, frame_t=0.3, dt=0.1)
    assert len(frame_events) == 0


def test_events_in_frame_30fps():
    """Test events_in_frame with 30 fps frame duration."""
    dt = 1/30  # ~0.0333s

    events = [
        {"t": 0.01, "impulse": 5.0},
        {"t": 0.05, "impulse": 3.0},
        {"t": 0.1, "impulse": 7.0},
    ]

    # First frame (0 to 1/30)
    frame_events = events_in_frame(events, frame_t=0.0, dt=dt)
    assert len(frame_events) == 1
    assert frame_events[0]["t"] == 0.01

    # Second frame (1/30 to 2/30)
    frame_events = events_in_frame(events, frame_t=1/30, dt=dt)
    assert len(frame_events) == 1
    assert frame_events[0]["t"] == 0.05

    # Third frame (2/30 to 3/30)
    frame_events = events_in_frame(events, frame_t=2/30, dt=dt)
    assert len(frame_events) == 0  # t=0.1 is in frame 3 (3/30 = 0.1)


@pytest.mark.integration
def test_load_episode_single_object(tmp_path):
    """Test loading a single-object episode from outputs directory."""
    # This test requires actual outputs from running the simulation
    # Check if outputs directory exists
    outputs_dir = Path("outputs")

    if not outputs_dir.exists():
        pytest.skip("No outputs directory found - run demo first")

    # Find first scenario directory
    scenario_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
    if not scenario_dirs:
        pytest.skip("No scenario outputs found")

    # Try to load first episode
    scenario_dir = scenario_dirs[0]

    episode = load_episode(scenario_dir)

    # Validate episode structure
    assert episode.name == scenario_dir.name
    assert episode.audio.shape[0] > 0  # Has audio samples
    assert episode.duration > 0
    assert isinstance(episode.telemetry, dict)
    assert isinstance(episode.events, list)

    # Check audio sample rate (duration should match audio length / 44100)
    expected_duration = len(episode.audio) / 44100.0
    assert abs(episode.duration - expected_duration) < 0.01

    # Check telemetry has expected keys for single-object
    if not episode.is_multi:
        assert "t" in episode.telemetry
        assert "fn" in episode.telemetry
        assert "vt" in episode.telemetry
        assert "z" in episode.telemetry


@pytest.mark.integration
def test_load_episode_spatial(tmp_path):
    """Test loading spatial (binaural) episode."""
    outputs_dir = Path("outputs")

    if not outputs_dir.exists():
        pytest.skip("No outputs directory found")

    # Look for spatial outputs (binaural)
    spatial_dirs = []
    for d in outputs_dir.iterdir():
        if d.is_dir():
            wav_files = list(d.glob("*.wav"))
            if wav_files:
                # Check if binaural (load and check shape)
                import soundfile as sf
                audio, sr = sf.read(wav_files[0])
                if audio.ndim == 2 and audio.shape[1] == 2:
                    spatial_dirs.append(d)

    if not spatial_dirs:
        pytest.skip("No spatial outputs found")

    episode = load_episode(spatial_dirs[0])

    assert episode.is_spatial
    assert episode.audio.ndim == 2
    assert episode.audio.shape[1] == 2  # Stereo


@pytest.mark.integration
def test_load_episode_multi_object(tmp_path):
    """Test loading multi-object episode."""
    outputs_dir = Path("outputs")

    if not outputs_dir.exists():
        pytest.skip("No outputs directory found")

    # Look for multi-object outputs (has registry.json)
    multi_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and list(d.glob("*_registry.json"))]

    if not multi_dirs:
        pytest.skip("No multi-object outputs found")

    episode = load_episode(multi_dirs[0])

    assert episode.is_multi
    assert episode.registry is not None
    assert len(episode.registry) > 0

    # Telemetry should be dict of dicts for multi-object
    assert isinstance(episode.telemetry, dict)


def test_load_episode_invalid_dir():
    """Test that loading from nonexistent directory raises error."""
    with pytest.raises(FileNotFoundError):
        load_episode(Path("/nonexistent/directory"))


def test_load_episode_missing_wav(tmp_path):
    """Test that loading from directory without .wav raises error."""
    # Create directory with telemetry but no audio
    scenario_dir = tmp_path / "test_scenario"
    scenario_dir.mkdir()

    # Create empty telemetry file
    import numpy as np
    telemetry = {"t": np.array([0.0, 0.1]), "fn": np.array([0.0, 1.0])}
    np.savez(scenario_dir / "test_telemetry.npz", **telemetry)

    with pytest.raises(FileNotFoundError, match="No .wav file found"):
        load_episode(scenario_dir)
