"""Tests for LeRobotDataset writer."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from physonic.export.writer import (
    chunk_audio,
    downsample_telemetry_single,
    downsample_telemetry_multi,
    LeRobotWriter
)
from physonic.export.episode import load_episode


def test_chunk_audio_mono():
    """Test audio chunking for mono signal at 30 fps."""
    # 2 seconds of mono audio @ 44.1kHz
    sr = 44100
    duration = 2.0
    audio = np.random.randn(int(sr * duration))

    fps = 30
    chunks = chunk_audio(audio, fps=fps, sr=sr)

    # Should produce 60 chunks (2 sec * 30 fps)
    assert len(chunks) == 60

    # Each chunk should have 1470 samples (44100 / 30)
    samples_per_frame = int(sr / fps)
    assert chunks[0].shape[0] == samples_per_frame

    # All chunks should be mono (1D)
    for chunk in chunks:
        assert chunk.ndim == 1


def test_chunk_audio_stereo():
    """Test audio chunking for stereo signal at 30 fps."""
    # 2 seconds of stereo audio @ 44.1kHz
    sr = 44100
    duration = 2.0
    audio = np.random.randn(int(sr * duration), 2)

    fps = 30
    chunks = chunk_audio(audio, fps=fps, sr=sr)

    # Should produce 60 chunks
    assert len(chunks) == 60

    # Each chunk should have shape (1470, 2)
    samples_per_frame = int(sr / fps)
    assert chunks[0].shape == (samples_per_frame, 2)


def test_chunk_audio_custom_fps():
    """Test audio chunking with custom fps."""
    sr = 44100
    duration = 1.0
    audio = np.random.randn(int(sr * duration))

    # Test 60 fps
    fps = 60
    chunks = chunk_audio(audio, fps=fps, sr=sr)

    assert len(chunks) == 60
    assert chunks[0].shape[0] == int(sr / fps)  # 735 samples


def test_downsample_telemetry_single():
    """Test downsampling single-object telemetry from 5kHz to 30 fps."""
    # Create 2.5 seconds of 5kHz telemetry
    sr = 5000
    duration = 2.5
    n_samples = int(sr * duration)

    tele = {
        "t": np.arange(0, duration, 1/sr),
        "fn": np.random.randn(n_samples),
        "vt": np.random.randn(n_samples),
        "z": np.linspace(1.0, 0.0, n_samples)  # Falling from 1m to 0m
    }

    fps = 30
    states = downsample_telemetry_single(tele, fps=fps)

    # Should produce 75 frames (2.5 sec * 30 fps)
    assert len(states) == 75

    # Each state should be [z, vz, fn, vt] - 4 floats
    assert states[0].shape == (4,)
    assert states[0].dtype == np.float32

    # Check z values are reasonable (descending)
    z_values = [s[0] for s in states]
    assert z_values[0] > z_values[-1]  # Falling


def test_downsample_telemetry_multi():
    """Test downsampling multi-object telemetry."""
    # Create simple 2-object scenario
    sr = 5000
    duration = 1.0
    n_samples = int(sr * duration)

    # Registry for 2 objects
    registry = {
        "sphere1": {
            "mass": 1.0,
            "material": "steel",
            "initial_pos": [0, 0, 1.0]
        },
        "sphere2": {
            "mass": 0.5,
            "material": "wood",
            "initial_pos": [0.5, 0, 1.0]
        }
    }

    # Telemetry for one contact pair
    tele = {
        ("sphere1", "sphere2"): {
            "t": np.arange(0, duration, 1/sr),
            "fn": np.random.randn(n_samples),
            "ft": np.random.randn(n_samples),
            "pos": np.random.randn(n_samples, 3),  # Contact position
            "fn_s": np.random.randn(n_samples)
        }
    }

    fps = 30
    states = downsample_telemetry_multi(tele, registry, fps=fps)

    # Should produce 30 frames (1 sec * 30 fps)
    assert len(states) == 30

    # Each state should be [6*N + 2] floats for N=2 objects
    # 6 per object (x, y, z, vx, vy, vz) + 2 aggregate forces (fn, ft)
    expected_size = 6 * 2 + 2
    assert states[0].shape == (expected_size,)
    assert states[0].dtype == np.float32


@pytest.mark.integration
def test_writer_basic(tmp_path):
    """Test LeRobotWriter with synthetic episode."""
    # Create synthetic episode
    from physonic.export.episode import Episode

    sr = 44100
    duration = 1.0

    # Mono audio
    audio = np.random.randn(int(sr * duration))

    # Single-object telemetry at 5kHz
    physics_sr = 5000
    n_physics = int(physics_sr * duration)

    telemetry = {
        "t": np.arange(0, duration, 1/physics_sr),
        "fn": np.random.randn(n_physics),
        "vt": np.random.randn(n_physics),
        "z": np.linspace(1.0, 0.0, n_physics)
    }

    # Events
    events = [
        {"t": 0.5, "impulse": 10.0, "peak_force": 50.0}
    ]

    episode = Episode(
        name="test_episode",
        spec=None,
        audio=audio,
        telemetry=telemetry,
        events=events,
        registry=None,
        duration=duration,
        is_spatial=False,
        is_multi=False
    )

    # Create writer
    output_dir = tmp_path / "test_dataset"
    writer = LeRobotWriter(output_dir=output_dir, fps=30, chunk_size=100)

    # Add episode
    writer.add_episode(episode)

    # Write and finalize
    writer.write()
    writer.finalize()

    # Verify directory structure
    assert (output_dir / "data").exists()
    assert (output_dir / "audio" / "binaural").exists()
    assert (output_dir / "meta").exists()

    # Verify Parquet file exists
    parquet_files = list((output_dir / "data").glob("*.parquet"))
    assert len(parquet_files) > 0

    # Verify info.json exists and has correct structure
    info_path = output_dir / "meta" / "info.json"
    assert info_path.exists()

    import json
    with open(info_path, 'r') as f:
        info = json.load(f)

    assert info["fps"] == 30
    assert info["num_episodes"] == 1
    assert info["total_frames"] == 30  # 1 sec * 30 fps
    assert "audio" in info
    assert "physics" in info


@pytest.mark.integration
def test_writer_parquet_schema(tmp_path):
    """Test that generated Parquet has correct schema."""
    from physonic.export.episode import Episode

    sr = 44100
    duration = 0.5

    # Stereo audio
    audio = np.random.randn(int(sr * duration), 2)

    # Telemetry
    physics_sr = 5000
    n_physics = int(physics_sr * duration)

    telemetry = {
        "t": np.arange(0, duration, 1/physics_sr),
        "fn": np.random.randn(n_physics),
        "vt": np.random.randn(n_physics),
        "z": np.ones(n_physics)
    }

    episode = Episode(
        name="test_stereo",
        spec=None,
        audio=audio,
        telemetry=telemetry,
        events=[],
        registry=None,
        duration=duration,
        is_spatial=True,
        is_multi=False
    )

    # Create and run writer
    output_dir = tmp_path / "test_schema"
    writer = LeRobotWriter(output_dir=output_dir, fps=30)
    writer.add_episode(episode)
    writer.write()
    writer.finalize()

    # Load Parquet and check schema
    parquet_file = list((output_dir / "data").glob("*.parquet"))[0]
    df = pd.read_parquet(parquet_file)

    # Check columns
    expected_columns = [
        "observation.state",
        "observation.audio.binaural",
        "observation.video.front",
        "observation.video.side",
        "observation.video.top",
        "ground_truth.events",
        "ground_truth.materials",
        "ground_truth.sound_sources",
        "episode_index",
        "frame_index",
        "timestamp"
    ]

    for col in expected_columns:
        assert col in df.columns

    # Check data types
    assert df["episode_index"].dtype == np.int64
    assert df["frame_index"].dtype == np.int64
    assert df["timestamp"].dtype == np.float64

    # Check state vector size (single-object should be 4 floats)
    first_state = df["observation.state"].iloc[0]
    assert len(first_state) == 4


@pytest.mark.integration
def test_writer_audio_chunks(tmp_path):
    """Test that audio chunks are saved correctly."""
    from physonic.export.episode import Episode

    sr = 44100
    duration = 0.5

    # Stereo audio with known pattern
    audio = np.ones((int(sr * duration), 2))

    telemetry = {
        "t": np.arange(0, duration, 1/5000),
        "fn": np.zeros(int(5000 * duration)),
        "vt": np.zeros(int(5000 * duration)),
        "z": np.ones(int(5000 * duration))
    }

    episode = Episode(
        name="test_audio",
        spec=None,
        audio=audio,
        telemetry=telemetry,
        events=[],
        registry=None,
        duration=duration,
        is_spatial=True,
        is_multi=False
    )

    output_dir = tmp_path / "test_audio_chunks"
    writer = LeRobotWriter(output_dir=output_dir, fps=30)
    writer.add_episode(episode)
    writer.write()

    # Check audio chunks were saved
    audio_dir = output_dir / "audio" / "binaural"
    audio_files = list(audio_dir.glob("*.flac"))

    # Should have 15 chunks (0.5 sec * 30 fps)
    assert len(audio_files) == 15

    # Load one chunk and verify it's stereo FLAC
    import soundfile as sf
    chunk_audio, chunk_sr = sf.read(audio_files[0])

    assert chunk_sr == sr
    assert chunk_audio.shape[1] == 2  # Stereo
    assert chunk_audio.shape[0] == int(sr / 30)  # 1470 samples


@pytest.mark.integration
def test_writer_with_real_episode(tmp_path):
    """Test writer with real episode from outputs."""
    outputs_dir = Path("outputs")

    if not outputs_dir.exists():
        pytest.skip("No outputs directory found - run demo first")

    # Find first scenario
    scenario_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
    if not scenario_dirs:
        pytest.skip("No scenario outputs found")

    # Load episode
    episode = load_episode(scenario_dirs[0])

    # Export to dataset
    output_dir = tmp_path / "test_real_export"
    writer = LeRobotWriter(output_dir=output_dir, fps=30)
    writer.add_episode(episode)
    writer.write()
    writer.finalize()

    # Verify structure
    assert (output_dir / "data").exists()
    assert (output_dir / "meta" / "info.json").exists()

    # Verify Parquet is loadable
    parquet_files = list((output_dir / "data").glob("*.parquet"))
    assert len(parquet_files) > 0

    df = pd.read_parquet(parquet_files[0])
    assert len(df) > 0

    # Verify audio chunks exist
    audio_files = list((output_dir / "audio" / "binaural").glob("*.flac"))
    assert len(audio_files) > 0
    assert len(audio_files) == len(df)  # One audio chunk per frame


@pytest.mark.integration
def test_writer_multiple_episodes(tmp_path):
    """Test writer with multiple episodes."""
    from physonic.export.episode import Episode

    # Create 3 synthetic episodes
    episodes = []
    for i in range(3):
        sr = 44100
        duration = 0.5

        audio = np.random.randn(int(sr * duration))

        telemetry = {
            "t": np.arange(0, duration, 1/5000),
            "fn": np.random.randn(int(5000 * duration)),
            "vt": np.random.randn(int(5000 * duration)),
            "z": np.ones(int(5000 * duration))
        }

        episode = Episode(
            name=f"test_episode_{i}",
            spec=None,
            audio=audio,
            telemetry=telemetry,
            events=[],
            registry=None,
            duration=duration,
            is_spatial=False,
            is_multi=False
        )
        episodes.append(episode)

    # Export all episodes
    output_dir = tmp_path / "test_multi"
    writer = LeRobotWriter(output_dir=output_dir, fps=30, chunk_size=20)

    for ep in episodes:
        writer.add_episode(ep)

    writer.write()
    writer.finalize()

    # Verify metadata
    import json
    with open(output_dir / "meta" / "info.json", 'r') as f:
        info = json.load(f)

    assert info["num_episodes"] == 3
    assert info["total_frames"] == 45  # 3 * 15 frames

    # Verify all episodes are in episode_names
    assert len(info["episode_names"]) == 3


@pytest.mark.integration
def test_writer_with_video_single_camera(tmp_path):
    """Test LeRobotWriter with video rendering (single camera)."""
    from physonic.export.episode import Episode

    sr = 44100
    duration = 0.5  # Short duration for faster test

    # Mono audio
    audio = np.random.randn(int(sr * duration))

    # Single-object telemetry
    physics_sr = 5000
    n_physics = int(physics_sr * duration)

    telemetry = {
        "t": np.arange(0, duration, 1/physics_sr),
        "fn": np.random.randn(n_physics),
        "vt": np.random.randn(n_physics),
        "z": np.linspace(1.0, 0.0, n_physics)
    }

    episode = Episode(
        name="drop_steel_on_glass",
        spec=None,
        audio=audio,
        telemetry=telemetry,
        events=[],
        registry=None,
        duration=duration,
        is_spatial=False,
        is_multi=False
    )

    # Create writer with video rendering enabled
    output_dir = tmp_path / "test_video"
    writer = LeRobotWriter(
        output_dir=output_dir,
        fps=30,
        chunk_size=100,
        include_video=True,
        video_cameras="front",
        video_width=320,  # Smaller for faster test
        video_height=240
    )

    # Add episode
    writer.add_episode(episode)
    writer.write()
    writer.finalize()

    # Verify video directory exists
    video_dir = output_dir / "videos"
    assert video_dir.exists()

    # Verify video file was created
    video_files = list(video_dir.glob("*.mp4"))
    assert len(video_files) == 1  # Single camera
    assert video_files[0].name == "episode_00000_front.mp4"

    # Verify video file is not empty
    assert video_files[0].stat().st_size > 0

    # Verify Parquet contains video path
    parquet_file = list((output_dir / "data").glob("*.parquet"))[0]
    df = pd.read_parquet(parquet_file)

    # Check video columns exist
    assert "observation.video.front" in df.columns
    assert "observation.video.side" in df.columns
    assert "observation.video.top" in df.columns

    # Front video should have path, others should be None
    assert df["observation.video.front"].iloc[0] == "videos/episode_00000_front.mp4"
    assert pd.isna(df["observation.video.side"].iloc[0])
    assert pd.isna(df["observation.video.top"].iloc[0])


@pytest.mark.integration
def test_writer_with_video_multi_camera(tmp_path):
    """Test LeRobotWriter with multi-camera video rendering."""
    from physonic.export.episode import Episode

    sr = 44100
    duration = 0.5

    audio = np.random.randn(int(sr * duration))

    physics_sr = 5000
    n_physics = int(physics_sr * duration)

    telemetry = {
        "t": np.arange(0, duration, 1/physics_sr),
        "fn": np.random.randn(n_physics),
        "vt": np.random.randn(n_physics),
        "z": np.linspace(1.0, 0.0, n_physics)
    }

    episode = Episode(
        name="drop_steel_on_glass",
        spec=None,
        audio=audio,
        telemetry=telemetry,
        events=[],
        registry=None,
        duration=duration,
        is_spatial=False,
        is_multi=False
    )

    # Create writer with multi-camera mode
    output_dir = tmp_path / "test_multicam"
    writer = LeRobotWriter(
        output_dir=output_dir,
        fps=30,
        chunk_size=100,
        include_video=True,
        video_cameras="multi",
        video_width=320,
        video_height=240
    )

    writer.add_episode(episode)
    writer.write()
    writer.finalize()

    # Verify all 3 videos were created
    video_dir = output_dir / "videos"
    video_files = sorted(video_dir.glob("*.mp4"))
    assert len(video_files) == 3

    video_names = [v.name for v in video_files]
    assert "episode_00000_front.mp4" in video_names
    assert "episode_00000_side.mp4" in video_names
    assert "episode_00000_top.mp4" in video_names

    # Verify all videos are not empty
    for video_file in video_files:
        assert video_file.stat().st_size > 0

    # Verify Parquet contains all video paths
    parquet_file = list((output_dir / "data").glob("*.parquet"))[0]
    df = pd.read_parquet(parquet_file)

    assert df["observation.video.front"].iloc[0] == "videos/episode_00000_front.mp4"
    assert df["observation.video.side"].iloc[0] == "videos/episode_00000_side.mp4"
    assert df["observation.video.top"].iloc[0] == "videos/episode_00000_top.mp4"


@pytest.mark.integration
def test_writer_video_disabled(tmp_path):
    """Test that video rendering is skipped when disabled."""
    from physonic.export.episode import Episode

    sr = 44100
    duration = 0.5

    audio = np.random.randn(int(sr * duration))

    physics_sr = 5000
    n_physics = int(physics_sr * duration)

    telemetry = {
        "t": np.arange(0, duration, 1/physics_sr),
        "fn": np.random.randn(n_physics),
        "vt": np.random.randn(n_physics),
        "z": np.ones(n_physics)
    }

    episode = Episode(
        name="test_no_video",
        spec=None,
        audio=audio,
        telemetry=telemetry,
        events=[],
        registry=None,
        duration=duration,
        is_spatial=False,
        is_multi=False
    )

    # Create writer with video disabled (default)
    output_dir = tmp_path / "test_no_video"
    writer = LeRobotWriter(
        output_dir=output_dir,
        fps=30,
        include_video=False
    )

    writer.add_episode(episode)
    writer.write()
    writer.finalize()

    # Verify video directory does not exist
    video_dir = output_dir / "videos"
    assert not video_dir.exists()

    # Verify Parquet has video columns but they're None
    parquet_file = list((output_dir / "data").glob("*.parquet"))[0]
    df = pd.read_parquet(parquet_file)

    assert "observation.video.front" in df.columns
    assert pd.isna(df["observation.video.front"].iloc[0])
    assert pd.isna(df["observation.video.side"].iloc[0])
    assert pd.isna(df["observation.video.top"].iloc[0])


@pytest.mark.integration
def test_writer_video_multiple_episodes(tmp_path):
    """Test video rendering with multiple episodes."""
    from physonic.export.episode import Episode

    # Create 2 episodes
    episodes = []
    for i in range(2):
        sr = 44100
        duration = 0.5

        audio = np.random.randn(int(sr * duration))

        telemetry = {
            "t": np.arange(0, duration, 1/5000),
            "fn": np.random.randn(int(5000 * duration)),
            "vt": np.random.randn(int(5000 * duration)),
            "z": np.ones(int(5000 * duration))
        }

        episode = Episode(
            name=f"drop_steel_on_glass",
            spec=None,
            audio=audio,
            telemetry=telemetry,
            events=[],
            registry=None,
            duration=duration,
            is_spatial=False,
            is_multi=False
        )
        episodes.append(episode)

    # Export with video
    output_dir = tmp_path / "test_multi_video"
    writer = LeRobotWriter(
        output_dir=output_dir,
        fps=30,
        include_video=True,
        video_cameras="front",
        video_width=320,
        video_height=240
    )

    for ep in episodes:
        writer.add_episode(ep)

    writer.write()
    writer.finalize()

    # Verify 2 video files were created
    video_dir = output_dir / "videos"
    video_files = sorted(video_dir.glob("*.mp4"))
    assert len(video_files) == 2
    assert video_files[0].name == "episode_00000_front.mp4"
    assert video_files[1].name == "episode_00001_front.mp4"

    # Verify Parquet has correct video paths for each episode
    parquet_file = list((output_dir / "data").glob("*.parquet"))[0]
    df = pd.read_parquet(parquet_file)

    # Episode 0 frames should reference episode_00000_front.mp4
    ep0_frames = df[df["episode_index"] == 0]
    assert all(ep0_frames["observation.video.front"] == "videos/episode_00000_front.mp4")

    # Episode 1 frames should reference episode_00001_front.mp4
    ep1_frames = df[df["episode_index"] == 1]
    assert all(ep1_frames["observation.video.front"] == "videos/episode_00001_front.mp4")
