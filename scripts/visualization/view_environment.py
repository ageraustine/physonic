"""Real-time 3D viewer for Physonic environments.

This script visualizes generated environments in 3D using MuJoCo rendering.
For interactive viewing on macOS, use: mjpython view_environment.py [args]
For static multi-angle rendering, use regular python3.

Usage:
    # View existing XML scene (interactive on Linux, static on macOS)
    python3 view_environment.py --xml outputs/demo_environments/data_center_maintenance_42.xml

    # Generate and view data center on-the-fly
    python3 view_environment.py --env data_center --scenario maintenance --layout hot_aisle

    # Generate with custom parameters
    python3 view_environment.py --env data_center --scenario failure --layout pod --num-racks 12

    # For interactive viewer on macOS, use mjpython:
    mjpython view_environment.py --xml outputs/demo_environments/data_center_maintenance_42.xml
"""

import argparse
import mujoco
import sys
import platform
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from physonic.environments.data_center import (
    generate_data_center_environment,
    DataCenterConfig,
    scenario_data_center_maintenance,
    scenario_data_center_failure,
    scenario_data_center_normal,
)
from physonic.environments.objects import build_scene_with_static_objects


def render_static_views(model, data, save_path: str = None):
    """Render multi-angle static views of the environment.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        save_path: Optional path to save image
    """
    # Create renderer (use 480x480 to fit default offscreen buffer)
    renderer = mujoco.Renderer(model, height=480, width=480)

    # Define camera views (closer for better visibility)
    views = [
        {"name": "Front", "azimuth": 0, "elevation": -15, "distance": 18},
        {"name": "Top", "azimuth": 0, "elevation": -90, "distance": 15},
        {"name": "Side", "azimuth": 90, "elevation": -15, "distance": 18},
        {"name": "3D View", "azimuth": 45, "elevation": -20, "distance": 18},
    ]

    # Calculate room center for lookat (at object height)
    # Estimate from model bounds
    lookat = np.array([10.0, 7.5, 1.0])  # Look at object height (1m above ground)

    # Render views
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    for i, view in enumerate(views):
        # Step physics
        mujoco.mj_step(model, data)

        # Set free camera position
        # Use the scene's free camera (cam)
        opt = mujoco.MjvOption()
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.trackbodyid = -1

        # Set view parameters
        cam.azimuth = view["azimuth"]
        cam.elevation = view["elevation"]
        cam.distance = view["distance"]
        cam.lookat[:] = lookat

        # Update scene with custom camera
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, renderer.scene)

        # Render
        pixels = renderer.render()

        # Display
        axes[i].imshow(pixels)
        axes[i].set_title(view["name"], fontsize=14, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle("Physonic Environment - Multi-Angle View", fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to: {save_path}")

    plt.show()


def view_xml_interactive(xml_path: str):
    """Open interactive MuJoCo viewer for existing XML file.

    Args:
        xml_path: Path to MuJoCo XML file

    Note: Requires mjpython on macOS
    """
    print(f"Loading scene from {xml_path}...")

    # Load model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print(f"Scene loaded:")
    print(f"  Bodies: {model.nbody}")
    print(f"  Geoms: {model.ngeom}")
    print(f"\nControls:")
    print("  - Mouse drag: Rotate camera")
    print("  - Mouse scroll: Zoom")
    print("  - Arrow keys: Pan camera")
    print("  - Ctrl+Mouse: Pan camera")
    print("  - ESC or close window: Exit")
    print("\nOpening interactive viewer...")

    # Try to launch interactive viewer
    try:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Set camera for better initial view
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -20
            viewer.cam.distance = 25
            viewer.cam.lookat[:] = [10, 7.5, 0]

            # Run simulation
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
    except RuntimeError as e:
        print(f"\nInteractive viewer not available: {e}")
        print("Falling back to static multi-angle rendering...")
        render_static_views(model, data)


def view_xml(xml_path: str, interactive: bool = False):
    """View MuJoCo XML file.

    Args:
        xml_path: Path to MuJoCo XML file
        interactive: If True, try interactive viewer (requires mjpython on macOS)
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print(f"Loading scene from {xml_path}...")
    print(f"Scene loaded:")
    print(f"  Bodies: {model.nbody}")
    print(f"  Geoms: {model.ngeom}")
    print()

    if interactive:
        view_xml_interactive(xml_path)
    else:
        print("Rendering multi-angle static views...")
        save_path = xml_path.replace(".xml", "_3d_view.png")
        render_static_views(model, data, save_path)


def view_environment(
    env_type: str,
    scenario: str,
    layout: str,
    num_racks: int,
    seed: int,
    interactive: bool = False,
):
    """Generate and view environment in 3D.

    Args:
        env_type: Environment type ("data_center")
        scenario: Scenario type ("maintenance", "failure", "normal")
        layout: Layout type ("hot_aisle", "pod", "perimeter")
        num_racks: Number of server racks
        seed: Random seed
        interactive: If True, try interactive viewer (requires mjpython on macOS)
    """
    print("=" * 60)
    print("Physonic 3D Environment Viewer")
    print("=" * 60)
    print(f"Environment: {env_type}")
    print(f"Scenario: {scenario}")
    print(f"Layout: {layout}")
    print(f"Seed: {seed}")
    print()

    # Generate environment
    if env_type == "data_center":
        # Use predefined scenario if available
        if scenario == "maintenance" and layout == "hot_aisle" and num_racks == 8:
            print("Using predefined maintenance scenario...")
            static_objects, continuous_sounds, alert_sounds = scenario_data_center_maintenance(
                room_size=(20, 15, 3),
                seed=seed
            )
            room_size = (20, 15, 3)
        elif scenario == "failure" and layout == "pod":
            print("Using predefined failure scenario...")
            static_objects, continuous_sounds, alert_sounds = scenario_data_center_failure(
                room_size=(20, 15, 3),
                seed=seed
            )
            room_size = (20, 15, 3)
        elif scenario == "normal" and layout == "perimeter":
            print("Using predefined normal scenario...")
            static_objects, continuous_sounds, alert_sounds = scenario_data_center_normal(
                room_size=(20, 15, 3),
                seed=seed
            )
            room_size = (20, 15, 3)
        else:
            print("Using custom configuration...")
            config = DataCenterConfig(
                layout_type=layout,
                room_size=(20, 15, 3),
                num_racks=num_racks,
                num_cooling_units=2,
                rack_utilization=0.7,
                scenario_type=scenario,
                seed=seed,
            )
            static_objects, continuous_sounds, alert_sounds = generate_data_center_environment(config)
            room_size = config.room_size
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

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
    print(f"  Alert events: {len(alert_sounds)}")

    # Generate MuJoCo XML
    print(f"\nGenerating MuJoCo scene...")
    xml = build_scene_with_static_objects(static_objects, floor_size=(room_size[0], room_size[1]))

    # Load model
    print(f"\nLoading MuJoCo model...")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print(f"Scene loaded:")
    print(f"  Bodies: {model.nbody}")
    print(f"  Geoms: {model.ngeom}")
    print()

    # Calculate lookat from room size (at object height)
    lookat = np.array([room_size[0]/2, room_size[1]/2, 1.0])  # Look at object height

    if interactive:
        print("Attempting interactive viewer (requires mjpython on macOS)...")
        try:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.azimuth = 45
                viewer.cam.elevation = -20
                viewer.cam.distance = max(room_size[0], room_size[1]) * 1.5
                viewer.cam.lookat[:] = lookat

                while viewer.is_running():
                    mujoco.mj_step(model, data)
                    viewer.sync()
        except RuntimeError as e:
            print(f"\nInteractive viewer not available: {e}")
            print("Falling back to static multi-angle rendering...")
            render_static_views(model, data)
    else:
        print("Rendering multi-angle static views...")
        # Update lookat for rendering
        views_config = [
            {"name": "Front", "azimuth": 0, "elevation": -15, "distance": max(room_size[0], room_size[1]) * 1.5},
            {"name": "Top", "azimuth": 0, "elevation": -90, "distance": max(room_size[0], room_size[1])},
            {"name": "Side", "azimuth": 90, "elevation": -15, "distance": max(room_size[0], room_size[1]) * 1.5},
            {"name": "3D View", "azimuth": 45, "elevation": -20, "distance": max(room_size[0], room_size[1]) * 1.5},
        ]

        # Create renderer (use 480x480 to fit default offscreen buffer)
        renderer = mujoco.Renderer(model, height=480, width=480)

        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        axes = axes.flatten()

        for i, view in enumerate(views_config):
            # Step physics
            mujoco.mj_step(model, data)

            # Set camera
            opt = mujoco.MjvOption()
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.fixedcamid = -1
            cam.trackbodyid = -1
            cam.azimuth = view["azimuth"]
            cam.elevation = view["elevation"]
            cam.distance = view["distance"]
            cam.lookat[:] = lookat

            # Update scene and render
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, renderer.scene)
            pixels = renderer.render()

            axes[i].imshow(pixels)
            axes[i].set_title(view["name"], fontsize=14, fontweight='bold')
            axes[i].axis('off')

        plt.suptitle(f"Data Center - {scenario.title()} ({layout})", fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = f"outputs/demo_environments/{env_type}_{scenario}_{seed}_3d_view.png"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to: {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="3D viewer for Physonic environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View existing XML (static multi-angle rendering)
  python3 view_environment.py --xml outputs/demo_environments/data_center_maintenance_42.xml

  # Generate and view data center
  python3 view_environment.py --scenario maintenance --layout hot_aisle

  # Custom configuration
  python3 view_environment.py --scenario failure --layout pod --num-racks 12 --seed 99

Note: On macOS, use 'mjpython view_environment.py --interactive' for real-time interactive viewer.
        """
    )

    # Option 1: Load existing XML
    parser.add_argument(
        "--xml",
        type=str,
        help="Path to existing MuJoCo XML file to view",
    )

    # Option 2: Generate environment on-the-fly
    parser.add_argument(
        "--env",
        type=str,
        choices=["data_center"],
        default="data_center",
        help="Environment type to generate",
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
        "--interactive",
        action="store_true",
        help="Try interactive viewer (requires mjpython on macOS, works on Linux)",
    )

    args = parser.parse_args()

    if args.xml:
        # View existing XML
        if not Path(args.xml).exists():
            print(f"Error: XML file not found: {args.xml}")
            return
        view_xml(args.xml, interactive=args.interactive)
    else:
        # Generate and view environment
        view_environment(
            env_type=args.env,
            scenario=args.scenario,
            layout=args.layout,
            num_racks=args.num_racks,
            seed=args.seed,
            interactive=args.interactive,
        )


if __name__ == "__main__":
    main()
