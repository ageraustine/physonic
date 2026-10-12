# Outputs Directory Structure

This directory contains all generated acoustic simulation outputs, organized by experiment type.

## Directory Organization

```
outputs/
├── mono/              # Mono audio outputs (baseline)
├── spatial/           # Binaural spatial audio (HRTF-rendered)
├── reverb/            # Spatial + room acoustics
├── multi/             # Multi-object scenarios
├── custom/            # Custom experiments
└── examples/          # Tutorial/example outputs (from example.py)
```

## Quick Reference

### Generate Outputs

```bash
# Mono audio (8 scenarios)
python3 -m physonic.run_demo --out outputs/mono --seed 0

# Binaural spatial audio
python3 -m physonic.run_demo --spatial --hrtf-path assets/hrtf/test_kemar --out outputs/spatial

# With room acoustics
python3 -m physonic.run_demo --spatial --room warehouse_large --out outputs/reverb

# Multi-object scenario
python3 -m physonic.run_demo_multi --scenario stack_collapse --out outputs/multi
```

### File Types

Each scenario generates:
- **`.wav`** - Audio waveform (mono or stereo @ 44.1kHz)
- **`_telemetry.npz`** - Physics data (contact forces, positions, velocities @ 5kHz)
- **`_events.json`** - Discrete impact events (onset times, impulses, materials)
- **`_registry.json`** - Object metadata (multi-object scenarios only)

### Export to Dataset

```bash
# Export any subdirectory to LeRobotDataset format
python3 -m physonic.export_dataset --input outputs/mono --output datasets/physonic_v1
```

## Naming Conventions

### Single-Object Scenarios
- `drop_{object}_on_{surface}.wav` - Drop scenarios
- `slide_{object}_on_{surface}.wav` - Slide scenarios

Suffixes:
- No suffix: Mono audio
- `_binaural.wav`: Spatial audio (HRTF)
- `_binaural_reverb.wav`: Spatial + room acoustics

### Multi-Object Scenarios
- `{scenario_name}.wav` - Full scenario audio
- `{scenario_name}_telemetry.npz` - Per-pair contact telemetry
- `{scenario_name}_events.json` - All collision events with object labels
- `{scenario_name}_registry.json` - Object properties and initial states

Available scenarios:
- `two_sphere_collision`
- `stack_collapse`
- `sphere_cascade`
- `tumble_box`
- `scatter_pile`
- `domino_chain`
- `rolling_cylinders`

## Cleanup

To remove all generated outputs:
```bash
rm -rf outputs/*
```

To remove specific experiment type:
```bash
rm -rf outputs/mono
rm -rf outputs/spatial
```

## Git Ignore

All files in `outputs/` are excluded from git by default (see `.gitignore`), except:
- This README.md
- `spectrograms.png` (if present)

This keeps the repository clean while allowing local experimentation.
