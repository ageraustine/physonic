"""Interactive demo for Physonic procedural environment generation.

This script demonstrates Phase 3C environment generation with:
- Procedural layout generation (grid-based collision avoidance)
- Static objects (server racks, cooling units, etc.)
- Sound synthesis (continuous equipment sounds + alerts)
- MuJoCo scene export

Compatible with:
- MuJoCo (scene visualization)
- Isaac Sim (future USD export)

Usage:
    python3 demo_environment_gen.py --env data_center --scenario maintenance
    python3 demo_environment_gen.py --env data_center --scenario failure --seed 42
    python3 demo_environment_gen.py --env data_center --layout pod --visualize
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path

from physonic.environments.data_center import (
    generate_data_center_environment,
    DataCenterConfig,
    scenario_data_center_maintenance,
    scenario_data_center_failure,
    scenario_data_center_normal,
)
from physonic.environments.objects import build_scene_with_static_objects
from physonic.synthesis.mixer import mix_acoustic_scene


def visualize_layout(static_objects, room_size, save_path: str):
    """Create top-down visualization of environment layout.

    Args:
        static_objects: List of StaticObjectSpec
        room_size: (width, depth, height) tuple
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw room bounds
    room_width, room_depth, _ = room_size
    ax.set_xlim(0, room_width)
    ax.set_ylim(0, room_depth)
    ax.set_aspect('equal')

    # Color mapping
    color_map = {
        "server_rack": "steelblue",
        "cooling_unit": "lightcoral",
        "pallet_stack": "saddlebrown",
        "shelf_unit": "gray",
    }

    # Draw objects
    for obj in static_objects:
        x, y, z = obj.position
        width, depth, height = obj.size

        # Determine color
        color = "gray"
        for key in color_map:
            if key in obj.object_type:
                color = color_map[key]
                break

        # Create rectangle
        rect = patches.Rectangle(
            (x - width/2, y - depth/2),
            width,
            depth,
            linewidth=1,
            edgecolor='black',
            facecolor=color,
            alpha=0.7 if not obj.emits_continuous_sound else 0.9
        )
        ax.add_patch(rect)

        # Add label
        label = obj.name.replace("_", " ").title()
        ax.text(x, y, label, ha='center', va='center', fontsize=7, rotation=0)

        # Add sound indicator
        if obj.emits_continuous_sound:
            circle = patches.Circle((x, y), 0.2, color='yellow', alpha=0.5)
            ax.add_patch(circle)

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Environment Layout (Top View)")
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='steelblue',
               markersize=10, label='Server Rack'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightcoral',
               markersize=10, label='Cooling Unit'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
               markersize=8, label='Emits Sound'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Physonic Environment Generation Demo")
    parser.add_argument(
        "--env",
        type=str,
        choices=["data_center"],
        default="data_center",
        help="Environment type (warehouse and supermarket coming soon)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["maintenance", "failure", "normal"],
        default="maintenance",
        help="Scenario type",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["hot_aisle", "pod", "perimeter"],
        default="hot_aisle",
        help="Layout type",
    )
    parser.add_argument(
        "--num-racks",
        type=int,
        default=8,
        help="Number of server racks",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate layout visualization",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/demo_environments",
        help="Output directory",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Physonic Environment Generation Demo")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Scenario: {args.scenario}")
    print(f"Layout: {args.layout}")
    print(f"Seed: {args.seed}")
    print()

    # Generate environment using predefined scenario or custom config
    if args.scenario == "maintenance" and args.layout == "hot_aisle" and args.num_racks == 8:
        print("Using predefined maintenance scenario...")
        static_objects, continuous_sounds, alert_sounds = scenario_data_center_maintenance(
            room_size=(20, 15, 3),
            seed=args.seed
        )
        room_size = (20, 15, 3)
    elif args.scenario == "failure" and args.layout == "pod":
        print("Using predefined failure scenario...")
        static_objects, continuous_sounds, alert_sounds = scenario_data_center_failure(
            room_size=(20, 15, 3),
            seed=args.seed
        )
        room_size = (20, 15, 3)
    else:
        print("Using custom configuration...")
        config = DataCenterConfig(
            layout_type=args.layout,
            room_size=(20, 15, 3),
            num_racks=args.num_racks,
            num_cooling_units=2,
            rack_utilization=0.7,
            scenario_type=args.scenario,
            seed=args.seed,
        )
        static_objects, continuous_sounds, alert_sounds = generate_data_center_environment(config)
        room_size = config.room_size

    print(f"\nGenerated Environment:")
    print(f"  Room size: {room_size[0]}m × {room_size[1]}m × {room_size[2]}m")
    print(f"  Static objects: {len(static_objects)}")

    # Count object types
    object_counts = {}
    for obj in static_objects:
        obj_type = obj.object_type
        object_counts[obj_type] = object_counts.get(obj_type, 0) + 1

    for obj_type, count in sorted(object_counts.items()):
        print(f"    - {obj_type}: {count}")

    print(f"  Continuous sounds: {len(continuous_sounds)}")
    for sound in continuous_sounds:
        print(f"    - {sound.source_id}: {sound.sound_type} @ {sound.fundamental_freq:.0f}Hz")

    print(f"  Alert events: {len(alert_sounds)}")
    for alert, trigger_time in alert_sounds:
        print(f"    - {alert.alert_type} at {trigger_time:.1f}s")

    # Generate MuJoCo scene XML
    print(f"\nGenerating MuJoCo scene...")
    xml = build_scene_with_static_objects(static_objects, floor_size=(room_size[0], room_size[1]))
    xml_path = output_dir / f"{args.env}_{args.scenario}_{args.seed}.xml"
    with open(xml_path, "w") as f:
        f.write(xml)
    print(f"  Saved to: {xml_path}")

    # Generate acoustic scene
    print(f"\nGenerating acoustic scene...")
    duration = 5.0
    acoustic_scene = mix_acoustic_scene(
        impact_sounds=[],  # No impacts in this demo
        continuous_sounds=continuous_sounds,
        alert_sounds=alert_sounds,
        duration=duration,
        continuous_level=0.3,
        alert_level=0.6,
    )

    audio_path = output_dir / f"{args.env}_{args.scenario}_{args.seed}.wav"
    sf.write(audio_path, acoustic_scene, 44100)
    print(f"  Saved audio to: {audio_path}")
    print(f"  Duration: {duration}s ({len(acoustic_scene)} samples)")

    # Visualize layout
    if args.visualize:
        print(f"\nGenerating layout visualization...")
        viz_path = output_dir / f"{args.env}_{args.scenario}_{args.seed}_layout.png"
        visualize_layout(static_objects, room_size, str(viz_path))
        print(f"  Saved visualization to: {viz_path}")

    # Save summary
    summary = {
        "environment": args.env,
        "scenario": args.scenario,
        "layout": args.layout,
        "seed": args.seed,
        "room_size": room_size,
        "num_objects": len(static_objects),
        "num_sounds": len(continuous_sounds),
        "num_alerts": len(alert_sounds),
        "object_counts": object_counts,
    }

    import json
    summary_path = output_dir / f"{args.env}_{args.scenario}_{args.seed}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")

    print("\n" + "=" * 60)
    print(f"Demo complete! Check {output_dir}/ for outputs:")
    print(f"  - MuJoCo scene XML: {xml_path.name}")
    print(f"  - Audio: {audio_path.name}")
    if args.visualize:
        print(f"  - Layout visualization: {viz_path.name}")
    print(f"  - Summary: {summary_path.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
