"""Generate different data center layouts using the LayoutGrid system.

This script demonstrates how to use the layout system to create
different data center configurations for training diversity.
"""

import numpy as np
from typing import List, Tuple

from physonic.environments.layout import LayoutGrid, LayoutConstraints
from physonic.environments.dynamic_scene import (
    DynamicObjectSpec,
    MovingRobotSpec,
    generate_complex_data_center,
    build_complex_scene_xml,
)
from physonic.synthesis.continuous import ContinuousSoundSpec


def generate_data_center_hot_aisle(
    room_size: Tuple[float, float, float] = (30, 25, 4),
    num_racks: int = 24,
    seed: int = 0,
) -> Tuple[List[DynamicObjectSpec], List[ContinuousSoundSpec], LayoutGrid]:
    """Generate data center with hot-aisle/cold-aisle layout using LayoutGrid.

    Hot-aisle/cold-aisle: Rows of racks facing each other with alternating
    airflow directions. Aisles are 1.2m (hot) and 2.0m (cold).

    Args:
        room_size: (width, depth, height) in meters
        num_racks: Number of server racks
        seed: Random seed

    Returns:
        Tuple of (dynamic_objects, continuous_sounds, layout_grid)
    """
    rng = np.random.default_rng(seed)
    room_width, room_depth, room_height = room_size

    # Create layout grid
    constraints = LayoutConstraints(
        min_spacing=0.3,
        wall_clearance=1.0,
        aisle_width=1.2,  # Hot aisle width
        max_density=0.4,
    )
    grid = LayoutGrid(
        room_size=(room_width, room_depth),
        grid_resolution=0.5,
        constraints=constraints,
    )

    dynamic_objects = []
    continuous_sounds = []

    # Rack specifications
    rack_size = (0.6, 1.0)  # width, depth (2D for grid)
    rack_height = 2.0

    # Place racks in rows (hot-aisle pattern)
    num_rows = 4
    racks_per_row = num_racks // num_rows

    start_y = 3.0
    row_spacing = 3.5  # 1.0m rack depth + 2.5m cold aisle

    rack_count = 0
    for row in range(num_rows):
        y_pos = start_y + row * row_spacing
        start_x = 2.5

        # Place row of racks
        positions = grid.place_row(
            start_position=(start_x, y_pos),
            object_size=rack_size,
            num_objects=racks_per_row,
            direction=(1, 0),  # Along X axis
            spacing=1.8,  # 0.6m rack + 1.2m aisle
        )

        for i, (x, y) in enumerate(positions):
            rack_count += 1
            z = 1.0 + rng.uniform(-0.05, 0.05)

            # 70% of racks emit sound
            emits_sound = (rack_count % 10 != 0)

            rack = DynamicObjectSpec(
                name=f"rack_{rack_count}",
                object_type="server_rack",
                position=(x, y, z),
                rotation=0.0,
                size=(rack_size[0], rack_size[1], rack_height),
                material="steel",
                mass=200.0,
                is_mobile=False,
                emits_continuous_sound=emits_sound,
            )
            dynamic_objects.append(rack)

            if emits_sound:
                sound = ContinuousSoundSpec(
                    source_id=f"rack_{rack_count}",
                    sound_type="server_hum",
                    fundamental_freq=120.0 + rng.uniform(-5, 5),
                    harmonics=[1.0, 0.5, 0.3, 0.15],
                    noise_amount=0.2,
                    modulation_freq=1.5,
                    modulation_depth=0.15,
                    position=(x, y, z),
                    volume=0.5 + rng.uniform(0, 0.2),
                )
                continuous_sounds.append(sound)

    # Add cooling units at ends of rows
    cooling_size = (0.8, 0.8)
    cooling_positions = [
        (room_width - 2.0, start_y, 0.75),
        (room_width - 2.0, start_y + row_spacing, 0.75),
        (2.0, start_y + row_spacing * 2, 0.75),
        (2.0, start_y + row_spacing * 3, 0.75),
    ]

    for i, (x, y, z) in enumerate(cooling_positions):
        if grid.can_place((x, y), cooling_size):
            grid.place_object((x, y), cooling_size)

            cooling = DynamicObjectSpec(
                name=f"cooling_{i+1}",
                object_type="cooling_unit",
                position=(x, y, z),
                rotation=0.0,
                size=(cooling_size[0], cooling_size[1], 1.5),
                material="steel",
                mass=300.0,
                is_mobile=False,
                emits_continuous_sound=True,
            )
            dynamic_objects.append(cooling)

            sound = ContinuousSoundSpec(
                source_id=f"cooling_{i+1}",
                sound_type="cooling_fan",
                fundamental_freq=180.0 + rng.uniform(-10, 10),
                harmonics=[1.0, 0.6, 0.4, 0.2],
                noise_amount=0.4,
                modulation_freq=3.0,
                modulation_depth=0.3,
                position=(x, y, z),
                volume=0.7,
            )
            continuous_sounds.append(sound)

    print(f"Hot-aisle layout:")
    print(f"  Placed {rack_count} racks in {num_rows} rows")
    print(f"  Added {len([d for d in dynamic_objects if 'cooling' in d.object_type])} cooling units")
    print(f"  Grid occupancy: {grid.get_occupancy():.1%}")

    return dynamic_objects, continuous_sounds, grid


def generate_data_center_pod(
    room_size: Tuple[float, float, float] = (30, 25, 4),
    num_pods: int = 4,
    racks_per_pod: int = 6,
    seed: int = 0,
) -> Tuple[List[DynamicObjectSpec], List[ContinuousSoundSpec], LayoutGrid]:
    """Generate data center with pod layout using LayoutGrid.

    Pod layout: Clusters of racks in groups (pods) with cooling units
    in the center of each pod. Wide aisles between pods.

    Args:
        room_size: (width, depth, height) in meters
        num_pods: Number of rack pods
        racks_per_pod: Number of racks per pod
        seed: Random seed

    Returns:
        Tuple of (dynamic_objects, continuous_sounds, layout_grid)
    """
    rng = np.random.default_rng(seed)
    room_width, room_depth, room_height = room_size

    # Create layout grid
    constraints = LayoutConstraints(
        min_spacing=0.5,
        wall_clearance=1.5,
        aisle_width=3.0,  # Wide aisles between pods
        max_density=0.35,
    )
    grid = LayoutGrid(
        room_size=(room_width, room_depth),
        grid_resolution=0.5,
        constraints=constraints,
    )

    dynamic_objects = []
    continuous_sounds = []

    rack_size = (0.6, 1.0)
    rack_height = 2.0

    # Pod layout: 2x2 grid of pods
    pod_positions = [
        (6, 5),   # Top-left
        (18, 5),  # Top-right
        (6, 15),  # Bottom-left
        (18, 15), # Bottom-right
    ]

    rack_count = 0
    for pod_idx in range(min(num_pods, len(pod_positions))):
        pod_x, pod_y = pod_positions[pod_idx]

        # Place racks in 3x2 grid within pod
        positions = grid.place_grid_pattern(
            start_position=(pod_x - 2, pod_y - 1.5),
            object_size=rack_size,
            num_rows=2,
            num_cols=3,
            row_spacing=2.5,
            col_spacing=1.5,
        )

        for x, y in positions:
            rack_count += 1
            z = 1.0 + rng.uniform(-0.05, 0.05)
            emits_sound = (rack_count % 10 != 0)

            rack = DynamicObjectSpec(
                name=f"rack_{rack_count}",
                object_type="server_rack",
                position=(x, y, z),
                rotation=rng.uniform(0, 0.2),  # Slight variation
                size=(rack_size[0], rack_size[1], rack_height),
                material="steel",
                mass=200.0,
                is_mobile=False,
                emits_continuous_sound=emits_sound,
            )
            dynamic_objects.append(rack)

            if emits_sound:
                sound = ContinuousSoundSpec(
                    source_id=f"rack_{rack_count}",
                    sound_type="server_hum",
                    fundamental_freq=120.0 + rng.uniform(-5, 5),
                    harmonics=[1.0, 0.5, 0.3, 0.15],
                    noise_amount=0.2,
                    modulation_freq=1.5,
                    modulation_depth=0.15,
                    position=(x, y, z),
                    volume=0.5 + rng.uniform(0, 0.2),
                )
                continuous_sounds.append(sound)

        # Add cooling unit at center of pod
        cooling_pos = (pod_x, pod_y)
        cooling_size = (0.8, 0.8)

        if grid.can_place(cooling_pos, cooling_size):
            grid.place_object(cooling_pos, cooling_size)

            cooling = DynamicObjectSpec(
                name=f"cooling_{pod_idx+1}",
                object_type="cooling_unit",
                position=(cooling_pos[0], cooling_pos[1], 0.75),
                rotation=0.0,
                size=(cooling_size[0], cooling_size[1], 1.5),
                material="steel",
                mass=300.0,
                is_mobile=False,
                emits_continuous_sound=True,
            )
            dynamic_objects.append(cooling)

            sound = ContinuousSoundSpec(
                source_id=f"cooling_{pod_idx+1}",
                sound_type="cooling_fan",
                fundamental_freq=180.0 + rng.uniform(-10, 10),
                harmonics=[1.0, 0.6, 0.4, 0.2],
                noise_amount=0.4,
                modulation_freq=3.0,
                modulation_depth=0.3,
                position=(cooling_pos[0], cooling_pos[1], 0.75),
                volume=0.7,
            )
            continuous_sounds.append(sound)

    print(f"Pod layout:")
    print(f"  Created {min(num_pods, len(pod_positions))} pods")
    print(f"  Placed {rack_count} racks total")
    print(f"  Added {len([d for d in dynamic_objects if 'cooling' in d.object_type])} cooling units")
    print(f"  Grid occupancy: {grid.get_occupancy():.1%}")

    return dynamic_objects, continuous_sounds, grid


if __name__ == "__main__":
    import mujoco

    print("Data Center Layout Generation Examples")
    print("=" * 70)

    # Example 1: Hot-aisle layout
    print("\n1. Hot-Aisle/Cold-Aisle Layout")
    print("-" * 70)
    dynamic_objs, sounds, grid = generate_data_center_hot_aisle(
        room_size=(30, 25, 4),
        num_racks=24,
        seed=42,
    )

    # Build and test MuJoCo scene
    from physonic.environments.dynamic_scene import build_complex_scene_xml

    xml = build_complex_scene_xml(
        dynamic_objects=dynamic_objs,
        moving_robots=[],
        floor_size=(30, 25)
    )

    # Verify scene loads
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        print(f"  MuJoCo scene created: {model.nbody} bodies, {model.ngeom} geoms")
        print(f"  Continuous sounds: {len(sounds)}")
    except Exception as e:
        print(f"  Error loading MuJoCo scene: {e}")

    # Example 2: Pod layout
    print("\n2. Pod Layout")
    print("-" * 70)
    dynamic_objs2, sounds2, grid2 = generate_data_center_pod(
        room_size=(30, 25, 4),
        num_pods=4,
        racks_per_pod=6,
        seed=42,
    )

    xml2 = build_complex_scene_xml(
        dynamic_objects=dynamic_objs2,
        moving_robots=[],
        floor_size=(30, 25)
    )

    try:
        model2 = mujoco.MjModel.from_xml_string(xml2)
        data2 = mujoco.MjData(model2)
        print(f"  MuJoCo scene created: {model2.nbody} bodies, {model2.ngeom} geoms")
        print(f"  Continuous sounds: {len(sounds2)}")
    except Exception as e:
        print(f"  Error loading MuJoCo scene: {e}")

    # Example 3: Compare with current generation
    print("\n3. Current generate_complex_data_center()")
    print("-" * 70)
    objs3, robots3, sounds3 = generate_complex_data_center(
        room_size=(30, 25, 4),
        num_racks=24,
        num_robots=2,
        seed=42,
    )
    print(f"  Dynamic objects: {len(objs3)}")
    print(f"  Moving robots: {len(robots3)}")
    print(f"  Continuous sounds: {len(sounds3)}")

    xml3 = build_complex_scene_xml(
        dynamic_objects=objs3,
        moving_robots=robots3,
        floor_size=(30, 25)
    )

    try:
        model3 = mujoco.MjModel.from_xml_string(xml3)
        data3 = mujoco.MjData(model3)
        print(f"  MuJoCo scene created: {model3.nbody} bodies, {model3.ngeom} geoms")
    except Exception as e:
        print(f"  Error loading MuJoCo scene: {e}")

    print("\n" + "=" * 70)
    print("Layout generation complete!")
    print("\nKey Differences:")
    print("  - Hot-aisle: Structured rows with proper aisle spacing")
    print("  - Pod: Clustered racks with central cooling")
    print("  - Current: Basic hot-aisle with hardcoded spacing")
    print("\nUse these layouts to provide training diversity for PPO agents!")
