"""PyArrow schema definitions for LeRobotDataset format."""

import pyarrow as pa

# Parquet schema for Physonic LeRobotDataset
# Adapted for physics-driven acoustic synthesis with audio + state observations
LEROBOT_SCHEMA = pa.schema([
    # Physics state vector (downsampled from 5kHz to fps)
    # Single-object: [z, vz, fn, vt] - 4 floats
    # Multi-object: [obj1_x, obj1_y, obj1_z, obj1_vx, obj1_vy, obj1_vz, ..., total_fn, total_ft] - 6N+2 floats
    ("observation.state", pa.list_(pa.float32())),

    # Audio chunk path (FLAC file synchronized with this frame)
    ("observation.audio.binaural", pa.string()),

    # Ground truth labels (JSON strings for flexibility)
    ("ground_truth.events", pa.string()),              # JSON list of events in this frame
    ("ground_truth.materials", pa.string()),           # JSON: {obj: material}
    ("ground_truth.sound_sources", pa.string()),       # JSON: [{name, pos, vel, material}]

    # Episode and frame metadata
    ("episode_index", pa.int64()),
    ("frame_index", pa.int64()),
    ("timestamp", pa.float64()),  # Seconds from episode start
])
