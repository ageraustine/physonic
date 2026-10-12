#!/usr/bin/env python3
"""Export Physonic outputs to LeRobotDataset format.

Converts simulation outputs (audio + telemetry + events) to Parquet-based
multi-modal dataset for robot training.

Usage:
    # Export all scenarios from outputs/
    python3 -m physonic.export_dataset --input outputs/ --output datasets/physonic_warehouse_v1

    # Export specific scenarios
    python3 -m physonic.export_dataset --input outputs/ --output datasets/test \\
        --scenarios drop_steel_on_steel,slide_wood_on_glass

    # Custom FPS
    python3 -m physonic.export_dataset --input outputs/ --output datasets/physonic_30fps --fps 30

    # Custom Parquet chunk size
    python3 -m physonic.export_dataset --input outputs/ --output datasets/physonic_v1 \\
        --chunk-size 500
"""

import argparse
import sys
from pathlib import Path

from physonic.export.episode import load_episode
from physonic.export.writer import LeRobotWriter


def discover_episodes(input_dir: Path, scenario_filter: list = None) -> list:
    """Discover all episode directories in input folder.

    Handles two formats:
    1. Subdirectory format: outputs/scenario_name/*.wav
    2. Flat format: outputs/*.wav (groups by scenario name from filename)

    Args:
        input_dir: Input directory containing scenario outputs
        scenario_filter: Optional list of scenario names to include

    Returns:
        List of (scenario_name, scenario_path) tuples
    """
    episodes = []

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        return episodes

    # Check if this is a flat directory (files in root) or subdirectory structure
    wav_files_root = list(input_dir.glob("*.wav"))

    if wav_files_root:
        # Flat format: group files by scenario name
        scenarios = {}

        for wav_file in wav_files_root:
            # Extract scenario name (e.g., "drop_steel_on_glass.wav" -> "drop_steel_on_glass")
            scenario_name = wav_file.stem

            # Check if corresponding telemetry exists
            telemetry_file = input_dir / f"{scenario_name}_telemetry.npz"
            if not telemetry_file.exists():
                continue

            # Apply filter if specified
            if scenario_filter and scenario_name not in scenario_filter:
                continue

            scenarios[scenario_name] = input_dir

        # Add scenarios (all point to same directory in flat format)
        for scenario_name in sorted(scenarios.keys()):
            episodes.append((scenario_name, input_dir))

    else:
        # Subdirectory format: find all subdirectories with .wav files
        for scenario_dir in sorted(input_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue

            # Check if this directory has required files
            wav_files = list(scenario_dir.glob("*.wav"))
            telemetry_files = list(scenario_dir.glob("*_telemetry.npz"))

            if not wav_files or not telemetry_files:
                continue

            scenario_name = scenario_dir.name

            # Apply filter if specified
            if scenario_filter and scenario_name not in scenario_filter:
                continue

            episodes.append((scenario_name, scenario_dir))

    return episodes


def main():
    ap = argparse.ArgumentParser(
        description="Export Physonic outputs to LeRobotDataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    ap.add_argument(
        "--input",
        required=True,
        help="Input directory with scenario outputs (e.g., outputs/)"
    )

    ap.add_argument(
        "--output",
        required=True,
        help="Output dataset directory (e.g., datasets/physonic_v1)"
    )

    ap.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frame rate for Parquet rows (default: 30)"
    )

    ap.add_argument(
        "--scenarios",
        help="Comma-separated list of scenarios to include (default: all)"
    )

    ap.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of rows per Parquet chunk (default: 1000)"
    )

    args = ap.parse_args()

    # Parse inputs
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    scenario_filter = None
    if args.scenarios:
        scenario_filter = [s.strip() for s in args.scenarios.split(',')]

    # Discover episodes
    print(f"Discovering episodes in {input_dir}...")
    episodes = discover_episodes(input_dir, scenario_filter)

    if not episodes:
        print("Error: No valid episodes found.", file=sys.stderr)
        print("Make sure the input directory contains scenario folders with .wav and _telemetry.npz files.", file=sys.stderr)
        return 1

    print(f"Found {len(episodes)} episodes to export:")
    for name, path in episodes:
        print(f"  - {name}")

    # Create writer
    print(f"\nInitializing LeRobotDataset writer...")
    print(f"  Output: {output_dir}")
    print(f"  FPS: {args.fps}")
    print(f"  Chunk size: {args.chunk_size}")

    writer = LeRobotWriter(
        output_dir=output_dir,
        fps=args.fps,
        chunk_size=args.chunk_size
    )

    # Process each episode
    print(f"\nProcessing episodes...")
    for i, (name, path) in enumerate(episodes, 1):
        print(f"[{i}/{len(episodes)}] Loading {name}...")

        try:
            # For flat format, pass scenario_name; for subdirectory format, path is the scenario dir
            # Check if path contains the .wav file directly (flat format)
            wav_in_dir = (path / f"{name}.wav").exists()

            if wav_in_dir:
                # Flat format
                episode = load_episode(path, scenario_name=name)
            else:
                # Subdirectory format
                episode = load_episode(path)

            print(f"  Duration: {episode.duration:.2f}s, Spatial: {episode.is_spatial}, Multi: {episode.is_multi}")

            writer.add_episode(episode)
            print(f"  Added to dataset")

        except Exception as e:
            print(f"  Error loading episode: {e}", file=sys.stderr)
            continue

    # Write remaining rows and finalize
    print(f"\nFinalizing dataset...")
    writer.write()
    writer.finalize()

    print(f"\nDataset export complete!")
    print(f"  Location: {output_dir}")
    print(f"  Episodes: {writer.episode_index}")
    print(f"  Total frames: {writer.total_frames}")
    print(f"\nVerify with:")
    print(f"  ls -R {output_dir}")
    print(f"  cat {output_dir}/meta/info.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
