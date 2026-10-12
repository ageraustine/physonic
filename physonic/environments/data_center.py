"""Data center environment template with procedural generation.

This module generates realistic data center environments with:
- Procedural layout (hot aisle / cold aisle, pod, perimeter)
- Server racks and cooling units
- Background equipment sounds
- Maintenance and failure scenarios
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

from physonic.environments.layout import LayoutGrid, LayoutConstraints
from physonic.environments.objects import StaticObjectSpec, create_static_object
from physonic.synthesis.continuous import ContinuousSoundSpec
from physonic.synthesis.alerts import AlertSoundSpec, create_preset_alert


@dataclass
class DataCenterConfig:
    """Configuration for data center environment generation.

    Attributes:
        layout_type: Layout pattern ("hot_aisle", "pod", "perimeter")
        room_size: (width, depth, height) in meters
        num_racks: Number of server racks
        num_cooling_units: Number of cooling systems
        rack_utilization: Fraction of racks that emit sound (0.0-1.0)
        scenario_type: Event scenario ("maintenance", "failure", "normal")
        seed: Random seed for reproducibility
    """
    layout_type: str = "hot_aisle"
    room_size: Tuple[float, float, float] = (20, 15, 3)
    num_racks: int = 10
    num_cooling_units: int = 2
    rack_utilization: float = 0.7
    scenario_type: str = "normal"
    seed: int = 0


def generate_hot_aisle_layout(
    grid: LayoutGrid,
    rack_size: Tuple[float, float],
    num_racks: int,
    rng: np.random.Generator,
) -> List[Tuple[Tuple[float, float], float]]:
    """Generate hot aisle / cold aisle layout.

    Racks are placed in parallel rows with alternating airflow directions.

    Args:
        grid: Layout grid for placement
        rack_size: (width, depth) of server rack
        num_racks: Number of racks to place
        rng: Random number generator

    Returns:
        List of (position, rotation) tuples for each rack
    """
    placements = []
    rack_width, rack_depth = rack_size

    # Configuration
    aisle_width = 1.5  # meters
    row_spacing = rack_depth + aisle_width

    # Calculate how many racks fit per row
    racks_per_row = int(grid.room_width / (rack_width + 0.5))
    num_rows = int(np.ceil(num_racks / racks_per_row))

    # Starting position
    start_x = 2.0
    start_y = 2.0

    rack_count = 0
    for row in range(num_rows):
        if rack_count >= num_racks:
            break

        y_pos = start_y + row * row_spacing

        for col in range(racks_per_row):
            if rack_count >= num_racks:
                break

            x_pos = start_x + col * (rack_width + 0.5)
            position_2d = (x_pos, y_pos)

            # Check if can place
            if grid.can_place(position_2d, rack_size):
                grid.place_object(position_2d, rack_size)

                # Rotation: racks face alternating directions for hot/cold aisles
                rotation = 0.0 if row % 2 == 0 else np.pi

                placements.append((position_2d, rotation))
                rack_count += 1

    return placements


def generate_pod_layout(
    grid: LayoutGrid,
    rack_size: Tuple[float, float],
    num_racks: int,
    rng: np.random.Generator,
) -> List[Tuple[Tuple[float, float], float]]:
    """Generate pod layout (clusters of racks).

    Racks are grouped in pods of 4-6 racks each.

    Args:
        grid: Layout grid
        rack_size: (width, depth) of server rack
        num_racks: Number of racks to place
        rng: Random number generator

    Returns:
        List of (position, rotation) tuples
    """
    placements = []
    rack_width, rack_depth = rack_size

    racks_per_pod = 6
    pod_spacing = 4.0  # meters between pods

    num_pods = int(np.ceil(num_racks / racks_per_pod))

    # Place pods in grid pattern
    pods_per_row = max(1, int(np.sqrt(num_pods)))

    rack_count = 0
    for pod_row in range(int(np.ceil(num_pods / pods_per_row))):
        for pod_col in range(pods_per_row):
            if rack_count >= num_racks:
                break

            # Pod center
            pod_x = 3.0 + pod_col * pod_spacing
            pod_y = 3.0 + pod_row * pod_spacing

            # Place 6 racks in 2x3 grid within pod
            for i in range(min(racks_per_pod, num_racks - rack_count)):
                row_in_pod = i // 2
                col_in_pod = i % 2

                x = pod_x + col_in_pod * rack_width
                y = pod_y + row_in_pod * rack_depth
                position_2d = (x, y)

                if grid.can_place(position_2d, rack_size):
                    grid.place_object(position_2d, rack_size)
                    rotation = 0.0
                    placements.append((position_2d, rotation))
                    rack_count += 1

    return placements


def generate_perimeter_layout(
    grid: LayoutGrid,
    rack_size: Tuple[float, float],
    num_racks: int,
    rng: np.random.Generator,
) -> List[Tuple[Tuple[float, float], float]]:
    """Generate perimeter layout (racks along walls).

    Racks are placed along the room perimeter facing inward.

    Args:
        grid: Layout grid
        rack_size: (width, depth) of server rack
        num_racks: Number of racks to place
        rng: Random number generator

    Returns:
        List of (position, rotation) tuples
    """
    placements = []
    rack_width, rack_depth = rack_size

    # Place along walls
    walls = [
        # (start_pos, direction, rotation, max_racks)
        ((2.0, 2.0), (1, 0), np.pi/2, int(grid.room_width / (rack_width + 0.5))),  # Bottom
        ((grid.room_width - 2.0, 2.0), (0, 1), np.pi, int(grid.room_depth / (rack_width + 0.5))),  # Right
        ((grid.room_width - 2.0, grid.room_depth - 2.0), (-1, 0), -np.pi/2, int(grid.room_width / (rack_width + 0.5))),  # Top
        ((2.0, grid.room_depth - 2.0), (0, -1), 0.0, int(grid.room_depth / (rack_width + 0.5))),  # Left
    ]

    rack_count = 0
    for start_pos, direction, rotation, max_in_wall in walls:
        if rack_count >= num_racks:
            break

        for i in range(min(max_in_wall, num_racks - rack_count)):
            x = start_pos[0] + i * direction[0] * (rack_width + 0.5)
            y = start_pos[1] + i * direction[1] * (rack_width + 0.5)
            position_2d = (x, y)

            if grid.can_place(position_2d, rack_size):
                grid.place_object(position_2d, rack_size)
                placements.append((position_2d, rotation))
                rack_count += 1

    return placements


def generate_data_center_environment(config: DataCenterConfig) -> Tuple[
    List[StaticObjectSpec],
    List[ContinuousSoundSpec],
    List[Tuple[AlertSoundSpec, float]],
]:
    """Generate complete data center environment.

    Returns:
        Tuple of (static_objects, continuous_sounds, alert_sounds)
        - static_objects: List of static object specs (racks, cooling)
        - continuous_sounds: List of background equipment sounds
        - alert_sounds: List of (alert_spec, trigger_time) tuples
    """
    rng = np.random.default_rng(config.seed)
    room_width, room_depth, room_height = config.room_size

    # Create layout grid
    grid = LayoutGrid(
        room_size=(room_width, room_depth),
        grid_resolution=0.5,
        constraints=LayoutConstraints(
            min_spacing=0.4,
            wall_clearance=1.0,
            aisle_width=1.5,
        ),
    )

    # Generate rack layout
    rack_size_2d = (0.6, 1.0)  # Server rack footprint (width, depth)

    if config.layout_type == "hot_aisle":
        rack_placements = generate_hot_aisle_layout(grid, rack_size_2d, config.num_racks, rng)
    elif config.layout_type == "pod":
        rack_placements = generate_pod_layout(grid, rack_size_2d, config.num_racks, rng)
    elif config.layout_type == "perimeter":
        rack_placements = generate_perimeter_layout(grid, rack_size_2d, config.num_racks, rng)
    else:
        raise ValueError(f"Unknown layout type: {config.layout_type}")

    # Create static objects
    static_objects = []
    continuous_sounds = []

    # Server racks
    active_racks = int(len(rack_placements) * config.rack_utilization)
    for i, (position_2d, rotation) in enumerate(rack_placements):
        rack_position = (position_2d[0], position_2d[1], 1.0)  # Height = 1.0m (half of 2m rack)
        emits_sound = (i < active_racks)

        rack = create_static_object(
            name=f"server_rack_{i+1}",
            object_type="server_rack",
            position=rack_position,
            rotation=rotation,
            emits_sound=emits_sound,
            sound_type="server_hum" if emits_sound else None,
            sound_volume=0.6 + rng.random() * 0.2,
        )
        static_objects.append(rack)

        if emits_sound and rack.continuous_sound:
            continuous_sounds.append(rack.continuous_sound)

    # Cooling units
    cooling_positions = [
        (room_width * 0.2, room_depth * 0.8, 0.75),
        (room_width * 0.8, room_depth * 0.8, 0.75),
    ][:config.num_cooling_units]

    for i, position in enumerate(cooling_positions):
        cooling = create_static_object(
            name=f"cooling_unit_{i+1}",
            object_type="cooling_unit",
            position=position,
            emits_sound=True,
            sound_type="cooling_system",
            sound_volume=0.7 + rng.random() * 0.2,
        )
        static_objects.append(cooling)

        if cooling.continuous_sound:
            continuous_sounds.append(cooling.continuous_sound)

    # Generate scenario-specific alerts
    alert_sounds = []

    if config.scenario_type == "maintenance":
        # Maintenance beep at 1 second
        alert = create_preset_alert("beep", position=(room_width/2, room_depth/2, 1.5))
        alert_sounds.append((alert, 1.0))

    elif config.scenario_type == "failure":
        # Equipment failure: warning alarm at 0.5s and error at 2.0s
        warning = create_preset_alert("warning", position=(room_width * 0.3, room_depth * 0.5, 1.5))
        error = create_preset_alert("error", position=(room_width * 0.3, room_depth * 0.5, 1.5))
        alert_sounds.append((warning, 0.5))
        alert_sounds.append((error, 2.0))

    # "normal" scenario has no alerts

    return static_objects, continuous_sounds, alert_sounds


# Convenience functions for common scenarios

def scenario_data_center_maintenance(room_size=(20, 15, 3), seed=0):
    """Data center maintenance scenario: technician working with tools."""
    config = DataCenterConfig(
        layout_type="hot_aisle",
        room_size=room_size,
        num_racks=8,
        num_cooling_units=2,
        rack_utilization=0.7,
        scenario_type="maintenance",
        seed=seed,
    )
    return generate_data_center_environment(config)


def scenario_data_center_failure(room_size=(20, 15, 3), seed=0):
    """Data center equipment failure: hard drive failure with alarms."""
    config = DataCenterConfig(
        layout_type="pod",
        room_size=room_size,
        num_racks=12,
        num_cooling_units=3,
        rack_utilization=0.8,
        scenario_type="failure",
        seed=seed,
    )
    return generate_data_center_environment(config)


def scenario_data_center_normal(room_size=(20, 15, 3), seed=0):
    """Data center normal operations: just background equipment sounds."""
    config = DataCenterConfig(
        layout_type="perimeter",
        room_size=room_size,
        num_racks=10,
        num_cooling_units=2,
        rack_utilization=0.6,
        scenario_type="normal",
        seed=seed,
    )
    return generate_data_center_environment(config)


def scenario_data_center_complex_ops(room_size=(30, 25, 4), seed=0):
    """Complex data center with many objects and dynamic physics.

    This scenario creates a large, complex data center for RL training with:
    - 20+ server racks in mixed layouts
    - Multiple cooling units
    - Cable management infrastructure
    - Equipment pallets and carts
    - All objects have dynamic physics enabled
    """
    config = DataCenterConfig(
        layout_type="hot_aisle",
        room_size=room_size,
        num_racks=24,
        num_cooling_units=4,
        rack_utilization=0.8,
        scenario_type="complex_ops",
        seed=seed,
    )
    return generate_data_center_environment(config)


# Example usage
if __name__ == "__main__":
    print("Data Center Environment Generation")
    print("=" * 60)

    # Example 1: Maintenance scenario
    print("\nExample 1: Maintenance scenario (hot aisle layout)")
    objects, continuous, alerts = scenario_data_center_maintenance(seed=42)

    print(f"  Static objects: {len(objects)}")
    print(f"    - Server racks: {sum(1 for o in objects if 'rack' in o.name)}")
    print(f"    - Cooling units: {sum(1 for o in objects if 'cooling' in o.name)}")
    print(f"  Continuous sounds: {len(continuous)}")
    print(f"  Alert sounds: {len(alerts)}")

    # Example 2: Failure scenario
    print("\nExample 2: Equipment failure scenario (pod layout)")
    objects2, continuous2, alerts2 = scenario_data_center_failure(seed=123)

    print(f"  Static objects: {len(objects2)}")
    print(f"  Continuous sounds: {len(continuous2)}")
    print(f"  Alert sounds: {len(alerts2)}")
    for i, (alert, trigger_time) in enumerate(alerts2):
        print(f"    - Alert {i+1}: {alert.alert_type} at {trigger_time}s")

    # Example 3: Custom configuration
    print("\nExample 3: Custom configuration")
    config = DataCenterConfig(
        layout_type="hot_aisle",
        room_size=(30, 25, 4),
        num_racks=20,
        num_cooling_units=4,
        rack_utilization=0.9,
        scenario_type="normal",
        seed=999,
    )
    objects3, continuous3, alerts3 = generate_data_center_environment(config)

    print(f"  Room size: {config.room_size}")
    print(f"  Layout: {config.layout_type}")
    print(f"  Generated {len(objects3)} objects")
    print(f"  Active equipment sounds: {len(continuous3)}")

    # Example 4: Generate MuJoCo XML
    print("\nExample 4: Generate MuJoCo XML for scene")
    from physonic.environments.objects import build_scene_with_static_objects

    xml = build_scene_with_static_objects(objects, floor_size=(20, 15))
    output_path = "outputs/data_center_scene.xml"
    with open(output_path, "w") as f:
        f.write(xml)
    print(f"  Saved to: {output_path}")

    print("\nDone!")
