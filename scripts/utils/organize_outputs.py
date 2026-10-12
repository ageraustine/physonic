#!/usr/bin/env python3
"""
Organize Physonic Outputs
==========================

Helper script to organize scattered output files into the standardized
outputs/ directory structure.

Usage:
    python3 scripts/organize_outputs.py

This will:
1. Find all .wav files in the project root and old output directories
2. Categorize them based on filename patterns
3. Move them to appropriate outputs/ subdirectories
4. Clean up empty directories
"""

import shutil
from pathlib import Path
import re


def categorize_file(filename: str) -> str:
    """Determine which output subdirectory a file belongs to."""
    if "binaural_reverb" in filename:
        return "reverb"
    elif "binaural" in filename:
        return "spatial"
    elif re.match(r"(drop|slide)_", filename):
        return "mono"
    else:
        # Multi-object scenarios (named after scenario)
        return "multi"


def find_output_files(root_dir: Path) -> list:
    """Find all Physonic output files."""
    patterns = ["*.wav", "*_telemetry.npz", "*_events.json", "*_registry.json", "summary.json"]

    files = []
    for pattern in patterns:
        # Search in root
        files.extend(root_dir.glob(pattern))

        # Search in old output directories
        for old_dir in root_dir.glob("outputs_*"):
            if old_dir.is_dir():
                files.extend(old_dir.glob(pattern))

    return files


def organize_outputs(dry_run: bool = True):
    """Organize output files into standard structure."""
    root = Path(__file__).parent.parent
    outputs_dir = root / "outputs"

    # Create subdirectories
    subdirs = ["mono", "spatial", "reverb", "multi", "custom"]
    for subdir in subdirs:
        (outputs_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Find files
    files = find_output_files(root)

    if not files:
        print("No output files found to organize.")
        return

    print(f"Found {len(files)} output files")
    print()

    # Group by category
    categorized = {}
    for file_path in files:
        if file_path.name == "spectrograms.png":
            continue  # Keep in outputs root

        if file_path.name == "summary.json":
            category = categorize_file(file_path.parent.name)
        else:
            category = categorize_file(file_path.stem)

        if category not in categorized:
            categorized[category] = []
        categorized[category].append(file_path)

    # Show plan
    for category, file_list in sorted(categorized.items()):
        print(f"{category}/ ({len(file_list)} files)")
        for f in sorted(file_list)[:3]:  # Show first 3
            print(f"  - {f.name}")
        if len(file_list) > 3:
            print(f"  ... and {len(file_list) - 3} more")
        print()

    if dry_run:
        print("=" * 60)
        print("DRY RUN - No files were moved")
        print("Run with --execute to actually move files")
        print("=" * 60)
        return

    # Move files
    print("Moving files...")
    moved_count = 0

    for category, file_list in categorized.items():
        dest_dir = outputs_dir / category

        for file_path in file_list:
            dest_path = dest_dir / file_path.name

            # Skip if already in correct location
            if file_path.parent == dest_dir:
                continue

            # Handle duplicates
            if dest_path.exists():
                print(f"  Warning: {dest_path.name} already exists, skipping")
                continue

            # Move file
            shutil.move(str(file_path), str(dest_path))
            moved_count += 1

    print(f"✓ Moved {moved_count} files")

    # Clean up empty old directories
    for old_dir in root.glob("outputs_*"):
        if old_dir.is_dir() and not any(old_dir.iterdir()):
            old_dir.rmdir()
            print(f"✓ Removed empty directory: {old_dir.name}")

    print()
    print("Organization complete!")
    print(f"Check outputs/ subdirectories for your files")


if __name__ == "__main__":
    import sys

    dry_run = "--execute" not in sys.argv

    print("Physonic Output Organizer")
    print("=" * 60)
    print()

    organize_outputs(dry_run=dry_run)
