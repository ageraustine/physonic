#!/usr/bin/env python3
"""
Physonic Quick Start Example
=============================

This script demonstrates the basic workflow:
1. Generate single-object acoustic scenarios (mono)
2. Generate with spatial audio (binaural)
3. Generate with room acoustics
4. Generate multi-object scenarios
5. Export to LeRobotDataset format

Run: python3 example.py
"""

import os
from pathlib import Path

print("=" * 60)
print("Physonic Quick Start Example")
print("=" * 60)

# Create outputs directory with subdirectories
outputs_dir = Path("outputs/examples")
outputs_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Step 1: Generate single-object scenarios (mono audio)
# ============================================================================
print("\n[1/5] Generating single-object scenarios (mono)...")
os.system(f"python3 -m physonic.run_demo --out {outputs_dir}/mono --seed 42")

print("\nGenerated files:")
mono_wavs = list((outputs_dir / "mono").glob("*.wav"))
print(f"  - {len(mono_wavs)} audio files (.wav)")
print(f"  - {len(list((outputs_dir / 'mono').glob('*_telemetry.npz')))} telemetry files")
print(f"  - {len(list((outputs_dir / 'mono').glob('*_events.json')))} event files")

# ============================================================================
# Step 2: Generate with spatial audio (binaural)
# ============================================================================
print("\n[2/5] Generating with spatial audio (binaural)...")

# Check if HRTF dataset is available
hrtf_path = Path("assets/hrtf/test_kemar")
if hrtf_path.exists():
    os.system(
        f"python3 -m physonic.run_demo --spatial --hrtf-path {hrtf_path} "
        f"--out {outputs_dir}/spatial --seed 42"
    )
    print("  ✓ Binaural audio generated")
else:
    print("  ⚠ HRTF dataset not found, skipping spatial audio")
    print("    Download MIT KEMAR HRTF and place in assets/hrtf/test_kemar/")

# ============================================================================
# Step 3: Generate with room acoustics
# ============================================================================
print("\n[3/5] Generating with room acoustics...")

if hrtf_path.exists():
    os.system(
        f"python3 -m physonic.run_demo --spatial --room warehouse_large "
        f"--hrtf-path {hrtf_path} --out {outputs_dir}/reverb --seed 42"
    )
    print("  ✓ Room acoustics applied (warehouse_large)")
else:
    print("  ⚠ Skipping (requires HRTF dataset)")

# ============================================================================
# Step 4: Generate multi-object scenario
# ============================================================================
print("\n[4/5] Generating multi-object scenario (stack collapse)...")

os.system(
    f"python3 -m physonic.run_demo_multi --scenario stack_collapse "
    f"--out {outputs_dir}/multi --seed 42"
)

multi_wavs = list((outputs_dir / "multi").glob("*.wav"))
print(f"  ✓ Generated {len(multi_wavs)} multi-object audio file(s)")

# ============================================================================
# Step 5: Export to LeRobotDataset
# ============================================================================
print("\n[5/5] Exporting to LeRobotDataset format...")

dataset_dir = Path("datasets/examples")
os.system(
    f"python3 -m physonic.export_dataset "
    f"--input {outputs_dir}/mono "
    f"--output {dataset_dir} "
    f"--fps 30"
)

print("\nDataset exported to:", dataset_dir)
print("  - data/chunk-000.parquet")
print("  - audio/binaural/*.flac")
print("  - meta/info.json")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("Example complete! 🎉")
print("=" * 60)
print("\nGenerated outputs in outputs/examples/:")
print(f"  1. Mono audio:       mono/")
if hrtf_path.exists():
    print(f"  2. Spatial audio:    spatial/")
    print(f"  3. Room acoustics:   reverb/")
else:
    print("  2. Spatial audio:    (skipped - download HRTF dataset)")
    print("  3. Room acoustics:   (skipped - download HRTF dataset)")
print(f"  4. Multi-object:     multi/")
print(f"\nDataset exported to: datasets/examples/")

print("\nNext steps:")
print("  - Listen to the generated audio files")
print("  - Inspect telemetry data: outputs/examples/mono/*_telemetry.npz")
print("  - Load dataset: pd.read_parquet('datasets/examples/data/chunk-000.parquet')")
print("  - See README.md for advanced usage")

print("\nClean up with:")
print("  rm -rf outputs/examples datasets/examples")
