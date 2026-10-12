"""Episode loading from Physonic outputs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import json
import numpy as np
import soundfile as sf


@dataclass
class Episode:
    """One simulation episode (single scenario run).

    Attributes:
        name: Episode identifier (e.g., "drop_steel_on_steel")
        spec: Scene specification (SceneSpec or MultiSceneSpec)
        audio: Audio waveform, shape (N,) mono or (N, 2) stereo @ 44.1kHz
        telemetry: Physics telemetry dictionary
            Single-object: {t, fn, vt, z} at 5kHz
            Multi-object: {(obj1, obj2): {t, fn, ft, pos, fn_s}}
        events: List of impact events with times, impulses, labels
        registry: Object registry (multi-object only), {obj_name: {mass, material, ...}}
        duration: Episode duration in seconds
        is_spatial: Whether audio is binaural (stereo) or mono
        is_multi: Whether this is a multi-object scenario
    """
    name: str
    spec: Optional[Any]  # SceneSpec or MultiSceneSpec (optional for export)
    audio: np.ndarray
    telemetry: Dict
    events: List[Dict]
    registry: Optional[Dict]
    duration: float
    is_spatial: bool
    is_multi: bool


def load_episode(scenario_dir: Path, scenario_name: str = None) -> Episode:
    """Load episode from outputs directory.

    Args:
        scenario_dir: Path to scenario outputs (e.g., outputs/drop_steel_on_steel/)
        scenario_name: Optional scenario name for flat directory format
                      (e.g., "drop_steel_on_glass" when all files are in same dir)

    Returns:
        Episode object with all data loaded

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If data is malformed
    """
    scenario_dir = Path(scenario_dir)

    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

    # Determine scenario name
    if scenario_name is None:
        # Subdirectory format: use directory name
        name = scenario_dir.name
        # Find audio file (any .wav)
        audio_files = list(scenario_dir.glob("*.wav"))
        if not audio_files:
            raise FileNotFoundError(f"No .wav file found in {scenario_dir}")
        audio_path = audio_files[0]
    else:
        # Flat format: use provided scenario name
        name = scenario_name
        # Find specific audio file
        audio_path = scenario_dir / f"{scenario_name}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sr = sf.read(audio_path)

    if sr != 44100:
        raise ValueError(f"Expected 44.1kHz audio, got {sr}Hz")

    is_spatial = (audio.ndim == 2 and audio.shape[1] == 2)

    # Load telemetry
    if scenario_name is None:
        # Subdirectory format: find any telemetry file
        telemetry_files = list(scenario_dir.glob("*_telemetry.npz"))
        if not telemetry_files:
            raise FileNotFoundError(f"No telemetry file found in {scenario_dir}")
        telemetry_path = telemetry_files[0]
    else:
        # Flat format: find specific telemetry file
        telemetry_path = scenario_dir / f"{scenario_name}_telemetry.npz"
        if not telemetry_path.exists():
            raise FileNotFoundError(f"Telemetry file not found: {telemetry_path}")

    telemetry_data = np.load(telemetry_path, allow_pickle=True)

    # Detect if multi-object (telemetry has 'pairs' key or is a dict of dicts)
    is_multi = False
    telemetry = {}

    if 'pairs' in telemetry_data:
        # Multi-object format with 'pairs' key
        is_multi = True
        pairs_data = telemetry_data['pairs'].item()  # Extract dict from 0-d array
        telemetry = pairs_data
    elif len(telemetry_data.files) > 0:
        # Check if it's single-object format (has t, fn, vt, z keys)
        if all(k in telemetry_data.files for k in ['t', 'fn', 'vt', 'z']):
            # Single-object format
            telemetry = {
                't': telemetry_data['t'],
                'fn': telemetry_data['fn'],
                'vt': telemetry_data['vt'],
                'z': telemetry_data['z']
            }
        else:
            # Try to interpret as multi-object dict
            is_multi = True
            # Reconstruct dict from npz
            for key in telemetry_data.files:
                telemetry[key] = telemetry_data[key]

    # Load events
    events = []
    if scenario_name is None:
        events_files = list(scenario_dir.glob("*_events.json"))
        if events_files:
            with open(events_files[0], 'r') as f:
                events = json.load(f)
    else:
        events_path = scenario_dir / f"{scenario_name}_events.json"
        if events_path.exists():
            with open(events_path, 'r') as f:
                events = json.load(f)

    # Load registry (multi-object only)
    registry = None
    if scenario_name is None:
        registry_files = list(scenario_dir.glob("*_registry.json"))
        if registry_files:
            is_multi = True
            with open(registry_files[0], 'r') as f:
                registry = json.load(f)
    else:
        registry_path = scenario_dir / f"{scenario_name}_registry.json"
        if registry_path.exists():
            is_multi = True
            with open(registry_path, 'r') as f:
                registry = json.load(f)

    # Reconstruct SceneSpec (simplified - we don't need full reconstruction for export)
    # For export purposes, we primarily need the telemetry/audio/events data
    spec = None  # We'll set this to None for now since we don't need it for basic export

    # Calculate duration
    duration = len(audio) / 44100.0

    return Episode(
        name=name,
        spec=spec,
        audio=audio,
        telemetry=telemetry,
        events=events,
        registry=registry,
        duration=duration,
        is_spatial=is_spatial,
        is_multi=is_multi
    )


def events_in_frame(events: List[Dict], frame_t: float, dt: float = 1/30) -> List[Dict]:
    """Return events that occurred during this frame.

    Args:
        events: List of event dictionaries with 't' key
        frame_t: Frame start time in seconds
        dt: Frame duration in seconds (default 1/30 for 30 fps)

    Returns:
        List of events occurring in [frame_t, frame_t + dt)
    """
    return [ev for ev in events if frame_t <= ev.get("t", 0) < frame_t + dt]
