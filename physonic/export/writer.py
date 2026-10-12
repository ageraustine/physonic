"""LeRobotDataset writer for Physonic episodes."""

from pathlib import Path
from typing import List, Dict, Optional
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from physonic.export.episode import Episode, events_in_frame
from physonic.export.schemas import LEROBOT_SCHEMA


def chunk_audio(audio: np.ndarray, fps: int = 30, sr: int = 44100) -> List[np.ndarray]:
    """Split audio into chunks aligned with video frames.

    Args:
        audio: Audio waveform, shape (N,) mono or (N, 2) stereo @ sample_rate
        fps: Target frame rate (default 30)
        sr: Audio sample rate (default 44100)

    Returns:
        List of audio chunks, one per frame
        Chunk i covers time [i/fps, (i+1)/fps)
    """
    samples_per_frame = int(sr / fps)  # 1470 samples @ 30 fps
    duration = len(audio) / sr
    n_frames = int(duration * fps)

    chunks = []
    for i in range(n_frames):
        start = i * samples_per_frame
        end = (i + 1) * samples_per_frame

        # Handle mono vs stereo
        if audio.ndim == 1:
            chunk = audio[start:end]
        else:
            chunk = audio[start:end, :]

        chunks.append(chunk)

    return chunks


def downsample_telemetry_single(tele: Dict, fps: int = 30) -> List[np.ndarray]:
    """Downsample 5kHz single-object telemetry to target fps.

    Args:
        tele: Telemetry dict with keys {t, fn, vt, z} at 5kHz
        fps: Target frame rate (default 30)

    Returns:
        List of state vectors [z, vz, fn, vt], one per frame
    """
    physics_times = tele["t"]  # 5kHz timestamps
    duration = physics_times[-1]
    frame_times = np.arange(0, duration, 1/fps)

    states = []
    for i, frame_t in enumerate(frame_times):
        # Find nearest physics timestep
        idx = np.argmin(np.abs(physics_times - frame_t))

        # Compute velocity (Δz/Δt)
        if idx > 0:
            dt = physics_times[idx] - physics_times[idx - 1]
            dz = tele["z"][idx] - tele["z"][idx - 1]
            vz = dz / dt if dt > 0 else 0.0
        else:
            vz = 0.0

        # State vector: [z, vz, fn, vt]
        state = np.array([
            tele["z"][idx],
            vz,
            tele["fn"][idx],
            tele["vt"][idx]
        ], dtype=np.float32)

        states.append(state)

    return states


def downsample_telemetry_multi(tele: Dict, registry: Dict, fps: int = 30) -> List[np.ndarray]:
    """Downsample 5kHz multi-object telemetry to target fps.

    Args:
        tele: Per-pair telemetry {(obj1, obj2): {t, fn, ft, pos, fn_s}}
        registry: Object registry {obj_name: {mass, material, initial_pos, ...}}
        fps: Target frame rate (default 30)

    Returns:
        List of state vectors [obj1_x, obj1_y, obj1_z, obj1_vx, obj1_vy, obj1_vz, ..., total_fn, total_ft]
    """
    # Get list of objects from registry
    obj_names = sorted(registry.keys())
    n_objects = len(obj_names)

    # Get representative time array (use first pair)
    first_pair = next(iter(tele.values()))
    physics_times = first_pair["t"]
    duration = physics_times[-1]
    frame_times = np.arange(0, duration, 1/fps)

    states = []
    for frame_t in frame_times:
        idx = np.argmin(np.abs(physics_times - frame_t))

        # Initialize state vector: 6 floats per object + 2 aggregate forces
        state = np.zeros(6 * n_objects + 2, dtype=np.float32)

        # Extract per-object positions (reconstruct from contact data)
        # Note: MuJoCo doesn't save full trajectory, we use contact positions as proxy
        # For objects in contact, use contact position; otherwise use initial position
        obj_positions = {}
        obj_velocities = {}

        for obj_name in obj_names:
            # Try to find contact position for this object
            found = False
            for (obj1, obj2), pair_tele in tele.items():
                if obj1 == obj_name or obj2 == obj_name:
                    # Use contact position
                    pos = pair_tele["pos"][idx]
                    obj_positions[obj_name] = pos

                    # Compute velocity from position derivative
                    if idx > 0:
                        dt = physics_times[idx] - physics_times[idx - 1]
                        dpos = pair_tele["pos"][idx] - pair_tele["pos"][idx - 1]
                        vel = dpos / dt if dt > 0 else np.zeros(3)
                    else:
                        vel = np.zeros(3)
                    obj_velocities[obj_name] = vel
                    found = True
                    break

            if not found:
                # No contact, use initial position
                initial_pos = registry[obj_name].get("initial_pos", [0, 0, 0])
                obj_positions[obj_name] = np.array(initial_pos)
                obj_velocities[obj_name] = np.zeros(3)

        # Fill per-object state
        for i, obj_name in enumerate(obj_names):
            pos = obj_positions.get(obj_name, np.zeros(3))
            vel = obj_velocities.get(obj_name, np.zeros(3))

            state[i*6 : i*6 + 3] = pos
            state[i*6 + 3 : i*6 + 6] = vel

        # Compute aggregate forces (sum over all pairs)
        total_fn = 0.0
        total_ft = 0.0

        for pair_tele in tele.values():
            total_fn += pair_tele["fn"][idx]
            if "ft" in pair_tele:
                total_ft += pair_tele["ft"][idx]

        state[-2] = total_fn
        state[-1] = total_ft

        states.append(state)

    return states


def downsample_telemetry(episode: Episode, fps: int = 30) -> List[np.ndarray]:
    """Downsample episode telemetry to target fps.

    Dispatches to single or multi-object downsampling based on episode type.

    Args:
        episode: Episode object
        fps: Target frame rate

    Returns:
        List of state vectors, one per frame
    """
    if episode.is_multi:
        return downsample_telemetry_multi(episode.telemetry, episode.registry, fps)
    else:
        return downsample_telemetry_single(episode.telemetry, fps)


class LeRobotWriter:
    """Converts Physonic episodes to LeRobotDataset format (Parquet + audio chunks).

    Usage:
        writer = LeRobotWriter(output_dir="datasets/physonic_v1", fps=30)
        writer.add_episode(episode1)
        writer.add_episode(episode2)
        writer.write()
        writer.finalize()
    """

    def __init__(self, output_dir: Path, fps: int = 30, chunk_size: int = 1000):
        """Initialize writer.

        Args:
            output_dir: Output dataset directory
            fps: Target frame rate for Parquet rows (default 30)
            chunk_size: Number of rows per Parquet chunk (default 1000)
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.chunk_size = chunk_size
        self.audio_sr = 44100

        # Create directory structure
        self.data_dir = self.output_dir / "data"
        self.audio_dir = self.output_dir / "audio" / "binaural"
        self.meta_dir = self.output_dir / "meta"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        # Accumulate rows before writing
        self.rows = []
        self.episode_index = 0
        self.global_frame_index = 0
        self.chunk_index = 0
        self.audio_chunk_index = 0

        # Metadata for info.json
        self.episode_names = []
        self.total_frames = 0
        self.materials = set()
        self.scenarios = {}

    def add_episode(self, episode: Episode):
        """Process one episode into frames at target fps.

        Args:
            episode: Episode object with audio, telemetry, events
        """
        # 1. Chunk audio into frame-aligned segments
        audio_chunks = chunk_audio(episode.audio, fps=self.fps, sr=self.audio_sr)

        # 2. Downsample telemetry to fps
        state_vectors = downsample_telemetry(episode, fps=self.fps)

        # Ensure audio and state have same number of frames
        n_frames = min(len(audio_chunks), len(state_vectors))

        # 3. Create Parquet rows
        for frame_idx in range(n_frames):
            frame_t = frame_idx / self.fps

            # Save audio chunk to FLAC
            audio_chunk = audio_chunks[frame_idx]
            audio_filename = f"chunk-{self.audio_chunk_index:05d}.flac"
            audio_path = self.audio_dir / audio_filename

            sf.write(audio_path, audio_chunk, self.audio_sr, format='FLAC')

            # Relative path for Parquet
            audio_rel_path = f"audio/binaural/{audio_filename}"

            # Ground truth: events in this frame
            frame_events = events_in_frame(episode.events, frame_t, dt=1/self.fps)
            events_json = json.dumps(frame_events)

            # Ground truth: materials
            if episode.is_multi and episode.registry:
                materials_dict = {
                    name: obj.get("material", "unknown")
                    for name, obj in episode.registry.items()
                }
                self.materials.update(materials_dict.values())
            else:
                # Single-object: extract from spec if available
                materials_dict = {}

            materials_json = json.dumps(materials_dict)

            # Ground truth: sound sources (multi-object only)
            sound_sources = []
            if episode.is_multi and episode.registry:
                # For each object, create sound source entry
                state_vec = state_vectors[frame_idx]
                n_objects = len(episode.registry)

                for i, (obj_name, obj_data) in enumerate(sorted(episode.registry.items())):
                    pos = state_vec[i*6 : i*6 + 3].tolist()
                    vel = state_vec[i*6 + 3 : i*6 + 6].tolist()

                    sound_sources.append({
                        "name": obj_name,
                        "pos": pos,
                        "vel": vel,
                        "material": obj_data.get("material", "unknown")
                    })

            sound_sources_json = json.dumps(sound_sources)

            # Create row
            row = {
                "observation.state": state_vectors[frame_idx].tolist(),
                "observation.audio.binaural": audio_rel_path,
                "ground_truth.events": events_json,
                "ground_truth.materials": materials_json,
                "ground_truth.sound_sources": sound_sources_json,
                "episode_index": self.episode_index,
                "frame_index": frame_idx,
                "timestamp": frame_t
            }

            self.rows.append(row)
            self.audio_chunk_index += 1
            self.global_frame_index += 1

            # Write Parquet chunk if we've accumulated enough rows
            if len(self.rows) >= self.chunk_size:
                self._write_chunk()

        # Update metadata
        self.episode_names.append(episode.name)
        self.total_frames += n_frames
        self.episode_index += 1

        # Track scenario types
        scenario_type = episode.name.split('_')[0]  # e.g., "drop", "slide", "stack"
        if scenario_type not in self.scenarios:
            self.scenarios[scenario_type] = episode.name

    def _write_chunk(self):
        """Write accumulated rows to Parquet file."""
        if not self.rows:
            return

        table = pa.Table.from_pylist(self.rows, schema=LEROBOT_SCHEMA)
        chunk_path = self.data_dir / f"chunk-{self.chunk_index:03d}.parquet"

        pq.write_table(table, chunk_path)

        print(f"Wrote {len(self.rows)} rows to {chunk_path.name}")

        self.rows = []
        self.chunk_index += 1

    def write(self):
        """Write any remaining rows to Parquet."""
        self._write_chunk()

    def finalize(self):
        """Write meta/info.json with dataset metadata."""
        info = {
            "codebase_version": "v1.0",
            "fps": self.fps,
            "audio": {
                "sample_rate": self.audio_sr,
                "channels": 2,
                "format": "FLAC",
                "chunk_duration_sec": 1 / self.fps
            },
            "physics": {
                "timestep": 0.0002,
                "sample_rate": 5000,
                "gravity": 9.81
            },
            "state_vector": {
                "single_object": ["z", "vz", "fn", "vt"],
                "multi_object": ["obj_x", "obj_y", "obj_z", "obj_vx", "obj_vy", "obj_vz", "total_fn", "total_ft"]
            },
            "num_episodes": self.episode_index,
            "total_frames": self.total_frames,
            "materials": sorted(list(self.materials)),
            "scenarios": self.scenarios,
            "episode_names": self.episode_names
        }

        info_path = self.meta_dir / "info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"Wrote metadata to {info_path}")
        print(f"Dataset summary: {self.episode_index} episodes, {self.total_frames} frames")
