"""Test USD export for photorealistic data center rendering."""

import numpy as np
from physonic.envs.data_center_nav import DataCenterNavigationEnv
from physonic.render.usd_exporter import USDSceneExporter


def test_usd_export():
    """Test USD export from data center environment."""
    print("=" * 70)
    print("Testing USD Export for Photorealistic Rendering")
    print("=" * 70)
    print()

    # Create data center environment
    print("Creating data center environment...")
    env = DataCenterNavigationEnv(
        room_size=(30, 25, 4),
        num_racks=24,
        use_mimi=True,  # Keep defaults
        max_steps=500,
        seed=42,
    )

    # Reset to get initial state
    print("Resetting environment...")
    try:
        obs, info = env.reset()
        print(f"Environment initialized: {env.num_racks} racks")
    except Exception as e:
        print(f"Note: Environment reset skipped ({e})")
        print("Using initial MuJoCo state for USD export...")
    print()

    # Create USD exporter
    print("Creating USD exporter...")
    exporter = USDSceneExporter(
        output_dir="outputs/usd_scenes",
        use_instancing=True,
    )

    # Export scene to USD
    print("Exporting scene to USD...")
    usd_path = exporter.export_scene(
        model=env.model,
        data=env.data,
        scene_name="datacenter_photorealistic",
        camera_position=(20, 15, 8),  # Elevated view
        camera_target=(15, 12.5, 0),  # Center of room
    )

    print()
    print("=" * 70)
    print("USD Export Complete!")
    print("=" * 70)
    print(f"Output file: {usd_path}")
    print()
    print("Next steps:")
    print("1. Isaac Sim (NVIDIA):")
    print("   - Open Isaac Sim")
    print(f"   - File > Open > {usd_path}")
    print("   - Real-time photorealistic rendering")
    print()
    print("2. Blender (Free):")
    print("   - Install USD addon")
    print(f"   - File > Import > Universal Scene Description (.usd/.usdc/.usda)")
    print("   - Select Cycles renderer for photorealistic output")
    print()
    print("3. Omniverse:")
    print(f"   - Drag {usd_path} into Omniverse viewport")
    print()


if __name__ == "__main__":
    test_usd_export()
