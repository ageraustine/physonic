"""Hyperscale data center environment for commercial robot training.

This module generates large-scale, realistic data centers like those in China:
- Multi-zone architecture (server halls, UPS rooms, cooling plants)
- 100s to 1000s of server racks
- Complex infrastructure (power, cooling, cable management)
- Realistic acoustic environment with zone-specific sound signatures
- Advanced robot training scenarios (patrol, emergency response, maintenance)

Designed for training commercial robots in real-world hyperscale facilities.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
from enum import Enum
import numpy as np

from physonic.environments.layout import LayoutGrid, LayoutConstraints
from physonic.environments.objects import StaticObjectSpec, create_static_object
from physonic.synthesis.continuous import ContinuousSoundSpec
from physonic.synthesis.alerts import AlertSoundSpec, create_preset_alert


class ZoneType(Enum):
    """Data center zone types."""
    SERVER_HALL = "server_hall"           # Main compute area
    STORAGE_ZONE = "storage_zone"         # Dense storage racks
    NETWORK_ZONE = "network_zone"         # Network equipment
    UPS_ROOM = "ups_room"                 # Power backup
    COOLING_PLANT = "cooling_plant"       # CRAC/CRAH units
    MAINTENANCE_AREA = "maintenance_area"  # Tools and parts
    HOT_AISLE = "hot_aisle"               # High-temperature exhaust
    COLD_AISLE = "cold_aisle"             # Cool air intake


class RackType(Enum):
    """Server rack types with different acoustic signatures."""
    COMPUTE = "server_rack"               # Standard compute servers
    STORAGE = "storage_rack"              # Storage arrays (disk noise)
    NETWORK = "network_rack"              # Network switches (fan noise)
    BLADE = "blade_server"                # High-density blades (louder fans)


@dataclass
class Zone:
    """A zone within the data center.

    Attributes:
        zone_type: Type of zone (server_hall, ups_room, etc.)
        bounds: (x_min, y_min, x_max, y_max) in meters
        rack_count: Number of racks in this zone
        rack_types: Distribution of rack types (if applicable)
        cooling_density: Cooling units per 10 racks
        temperature: Average temperature in °C (affects fan speed/noise)
        access_restricted: Whether robots need authorization to enter
    """
    zone_type: ZoneType
    bounds: Tuple[float, float, float, float]
    rack_count: int = 0
    rack_types: Dict[RackType, float] = field(default_factory=dict)  # Type -> probability
    cooling_density: float = 0.2  # Cooling units per 10 racks
    temperature: float = 22.0  # °C
    access_restricted: bool = False


@dataclass
class HyperscaleConfig:
    """Configuration for hyperscale data center generation.

    Attributes:
        name: Data center name (e.g., "Beijing_DC1", "Shenzhen_Cloud_3")
        total_floor_size: (width, depth) in meters (e.g., 100m x 80m)
        ceiling_height: Height in meters
        num_zones: Number of server hall zones
        racks_per_zone: Target racks per server hall
        layout_type: Layout pattern ("hot_aisle", "pod", "perimeter", "modular")
        include_support_rooms: Include UPS, cooling plant, maintenance areas
        power_density: kW per rack (affects cooling requirements)
        scenario_type: Training scenario type
        seed: Random seed
    """
    name: str = "Hyperscale_DC1"
    total_floor_size: Tuple[float, float] = (100.0, 80.0)
    ceiling_height: float = 4.0
    num_zones: int = 4
    racks_per_zone: int = 250
    layout_type: str = "hot_aisle"
    include_support_rooms: bool = True
    power_density: float = 8.0  # kW per rack
    scenario_type: str = "patrol"
    seed: int = 0


def calculate_zone_layout(
    total_floor_size: Tuple[float, float],
    num_server_zones: int,
    include_support_rooms: bool
) -> List[Zone]:
    """Calculate multi-zone layout for hyperscale data center.

    Creates a realistic layout with:
    - Multiple server halls
    - UPS room (10% of floor space)
    - Cooling plant (8% of floor space)
    - Maintenance area (5% of floor space)
    - Hot/cold aisle containment

    Args:
        total_floor_size: (width, depth) total floor area
        num_server_zones: Number of main server hall zones
        include_support_rooms: Whether to include UPS, cooling, maintenance areas

    Returns:
        List of Zone objects with calculated bounds
    """
    width, depth = total_floor_size
    zones = []

    if include_support_rooms:
        # Reserve space for support rooms (along one edge)
        support_width = width * 0.15  # 15% of width for support
        server_width = width - support_width

        # UPS room (top section)
        ups_height = depth * 0.35
        zones.append(Zone(
            zone_type=ZoneType.UPS_ROOM,
            bounds=(server_width, 0, width, ups_height),
            rack_count=0,
            access_restricted=True
        ))

        # Cooling plant (middle section)
        cooling_start = ups_height
        cooling_height = depth * 0.45
        zones.append(Zone(
            zone_type=ZoneType.COOLING_PLANT,
            bounds=(server_width, cooling_start, width, cooling_start + cooling_height),
            rack_count=0,
            access_restricted=False
        ))

        # Maintenance area (bottom section)
        maint_start = cooling_start + cooling_height
        zones.append(Zone(
            zone_type=ZoneType.MAINTENANCE_AREA,
            bounds=(server_width, maint_start, width, depth),
            rack_count=0,
            access_restricted=False
        ))
    else:
        server_width = width

    # Divide remaining space into server hall zones
    zone_depth = depth / num_server_zones

    for i in range(num_server_zones):
        y_start = i * zone_depth
        y_end = (i + 1) * zone_depth

        zones.append(Zone(
            zone_type=ZoneType.SERVER_HALL,
            bounds=(0, y_start, server_width, y_end),
            rack_count=0,  # Will be calculated during layout
            rack_types={
                RackType.COMPUTE: 0.6,
                RackType.STORAGE: 0.25,
                RackType.NETWORK: 0.10,
                RackType.BLADE: 0.05,
            },
            cooling_density=0.25,  # 1 cooling unit per 4 racks
            temperature=22.0 + np.random.uniform(-2, 2),
            access_restricted=False
        ))

    return zones


def generate_hot_aisle_hyperscale(
    zone: Zone,
    rack_size: Tuple[float, float],
    target_racks: int,
    rng: np.random.Generator,
    grid: LayoutGrid,
    zone_offset: Tuple[float, float] = (0, 0)
) -> List[Tuple[Tuple[float, float], float, RackType]]:
    """Generate hot aisle layout within a zone (hyperscale version).

    Creates high-density rows with proper aisle spacing for cooling efficiency.

    Args:
        zone: Zone to populate
        rack_size: (width, depth) of server rack
        target_racks: Target number of racks
        rng: Random number generator
        grid: Layout grid (for collision detection)
        zone_offset: (x_offset, y_offset) for zone position

    Returns:
        List of (position, rotation, rack_type) tuples
    """
    placements = []
    rack_width, rack_depth = rack_size

    # Calculate zone dimensions
    x_min, y_min, x_max, y_max = zone.bounds
    zone_width = x_max - x_min
    zone_depth = y_max - y_min

    # High-density configuration
    aisle_width = 1.2  # Narrow aisles for efficiency
    row_spacing = rack_depth + aisle_width
    rack_spacing = 0.3  # Tight spacing between racks in row

    # Calculate layout
    racks_per_row = int((zone_width - 2.0) / (rack_width + rack_spacing))
    num_rows = int((zone_depth - 2.0) / row_spacing)

    # Starting position within zone
    start_x = x_min + 1.0 + zone_offset[0]
    start_y = y_min + 1.0 + zone_offset[1]

    rack_count = 0
    rack_type_probs = list(zone.rack_types.values()) if zone.rack_types else [1.0]
    rack_type_choices = list(zone.rack_types.keys()) if zone.rack_types else [RackType.COMPUTE]

    for row in range(num_rows):
        if rack_count >= target_racks:
            break

        y_pos = start_y + row * row_spacing

        for col in range(racks_per_row):
            if rack_count >= target_racks:
                break

            x_pos = start_x + col * (rack_width + rack_spacing)
            position_2d = (x_pos, y_pos)

            # Check placement (with zone-local grid)
            if grid.can_place(position_2d, rack_size):
                grid.place_object(position_2d, rack_size)

                # Rotation for hot/cold aisle
                rotation = 0.0 if row % 2 == 0 else np.pi

                # Select rack type based on zone distribution
                rack_type = rng.choice(rack_type_choices, p=rack_type_probs)

                placements.append((position_2d, rotation, rack_type))
                rack_count += 1

    return placements


def generate_hyperscale_environment(config: HyperscaleConfig) -> Tuple[
    List[StaticObjectSpec],
    List[ContinuousSoundSpec],
    List[Tuple[AlertSoundSpec, float]],
    Dict[str, any]
]:
    """Generate complete hyperscale data center environment.

    Returns:
        Tuple of (static_objects, continuous_sounds, alert_sounds, metadata)
        - static_objects: List of all static objects (racks, cooling, infrastructure)
        - continuous_sounds: Background equipment sounds
        - alert_sounds: Alert events with trigger times
        - metadata: Zone information, stats, layout details
    """
    rng = np.random.default_rng(config.seed)
    width, depth = config.total_floor_size

    # Calculate zones
    zones = calculate_zone_layout(
        config.total_floor_size,
        config.num_zones,
        config.include_support_rooms
    )

    # Create global layout grid
    grid = LayoutGrid(
        room_size=(width, depth),
        grid_resolution=0.5,
        constraints=LayoutConstraints(
            min_spacing=0.3,
            wall_clearance=0.8,
            aisle_width=1.2,
        ),
    )

    static_objects = []
    continuous_sounds = []
    total_racks = 0
    total_cooling = 0
    total_power = 0

    rack_size_2d = (0.6, 1.0)  # Standard rack footprint

    # Process each zone
    for zone_idx, zone in enumerate(zones):
        if zone.zone_type == ZoneType.SERVER_HALL:
            # Generate racks
            rack_placements = generate_hot_aisle_hyperscale(
                zone,
                rack_size_2d,
                config.racks_per_zone,
                rng,
                grid
            )

            # Create rack objects
            for rack_idx, (position_2d, rotation, rack_type) in enumerate(rack_placements):
                rack_position = (position_2d[0], position_2d[1], 1.0)

                # 85% of racks are active and emit sound
                emits_sound = rng.random() < 0.85

                # Sound characteristics vary by rack type
                if rack_type == RackType.STORAGE:
                    sound_type = "equipment_buzz"  # Disk arrays have buzzing
                    sound_volume = 0.7 + rng.random() * 0.2
                elif rack_type == RackType.NETWORK:
                    sound_type = "fan"  # Network switches have fan noise
                    sound_volume = 0.5 + rng.random() * 0.15
                elif rack_type == RackType.BLADE:
                    sound_type = "cooling_system"  # Blade servers run hot
                    sound_volume = 0.8 + rng.random() * 0.15
                else:  # COMPUTE
                    sound_type = "server_hum"
                    sound_volume = 0.6 + rng.random() * 0.2

                rack_name = f"zone{zone_idx}_{rack_type.value}_{rack_idx}"

                rack = create_static_object(
                    name=rack_name,
                    object_type=rack_type.value,
                    position=rack_position,
                    rotation=rotation,
                    emits_sound=emits_sound,
                    sound_type=sound_type if emits_sound else None,
                    sound_volume=sound_volume,
                )
                static_objects.append(rack)
                total_racks += 1

                if emits_sound and rack.continuous_sound:
                    continuous_sounds.append(rack.continuous_sound)

            # Add cooling units (based on cooling_density)
            num_cooling = int(len(rack_placements) * zone.cooling_density)
            zone_bounds = zone.bounds
            zone_center_x = (zone_bounds[0] + zone_bounds[2]) / 2
            zone_center_y = (zone_bounds[1] + zone_bounds[3]) / 2

            for cool_idx in range(num_cooling):
                # Distribute cooling units around zone perimeter
                angle = (cool_idx / num_cooling) * 2 * np.pi
                offset_x = (zone_bounds[2] - zone_bounds[0]) * 0.4 * np.cos(angle)
                offset_y = (zone_bounds[3] - zone_bounds[1]) * 0.4 * np.sin(angle)

                cooling_pos = (
                    zone_center_x + offset_x,
                    zone_center_y + offset_y,
                    0.75
                )

                cooling = create_static_object(
                    name=f"zone{zone_idx}_cooling_{cool_idx}",
                    object_type="cooling_unit",
                    position=cooling_pos,
                    emits_sound=True,
                    sound_type="cooling_system",
                    sound_volume=0.75 + rng.random() * 0.2,
                )
                static_objects.append(cooling)
                total_cooling += 1

                if cooling.continuous_sound:
                    continuous_sounds.append(cooling.continuous_sound)

            # Add PDUs (power distribution) - one per row
            num_pdu_rows = int(np.sqrt(len(rack_placements) / 10))
            for pdu_idx in range(num_pdu_rows):
                pdu_y = zone_bounds[1] + (pdu_idx + 1) * (zone_bounds[3] - zone_bounds[1]) / (num_pdu_rows + 1)
                pdu_pos = (zone_bounds[2] - 1.0, pdu_y, 1.0)

                pdu = create_static_object(
                    name=f"zone{zone_idx}_pdu_{pdu_idx}",
                    object_type="pdu",
                    position=pdu_pos,
                    emits_sound=True,
                    sound_type="equipment_buzz",
                    sound_volume=0.3 + rng.random() * 0.1,
                )
                static_objects.append(pdu)
                total_power += 1

                if pdu.continuous_sound:
                    continuous_sounds.append(pdu.continuous_sound)

        elif zone.zone_type == ZoneType.UPS_ROOM:
            # UPS units (large power backup)
            zone_bounds = zone.bounds
            num_ups = 6

            for ups_idx in range(num_ups):
                ups_x = zone_bounds[0] + 2.0 + (ups_idx % 3) * 3.0
                ups_y = zone_bounds[1] + 2.0 + (ups_idx // 3) * 4.0

                ups = create_static_object(
                    name=f"ups_{ups_idx}",
                    object_type="ups_unit",
                    position=(ups_x, ups_y, 0.9),
                    emits_sound=True,
                    sound_type="equipment_buzz",
                    sound_volume=0.7 + rng.random() * 0.2,
                )
                static_objects.append(ups)

                if ups.continuous_sound:
                    continuous_sounds.append(ups.continuous_sound)

        elif zone.zone_type == ZoneType.COOLING_PLANT:
            # Large CRAC units
            zone_bounds = zone.bounds
            num_crac = 8

            for crac_idx in range(num_crac):
                crac_x = zone_bounds[0] + 2.0 + (crac_idx % 4) * 2.5
                crac_y = zone_bounds[1] + 2.0 + (crac_idx // 4) * 3.0

                crac = create_static_object(
                    name=f"crac_{crac_idx}",
                    object_type="crac_unit",
                    position=(crac_x, crac_y, 1.25),
                    emits_sound=True,
                    sound_type="hvac",
                    sound_volume=0.85 + rng.random() * 0.15,
                )
                static_objects.append(crac)

                if crac.continuous_sound:
                    continuous_sounds.append(crac.continuous_sound)

        elif zone.zone_type == ZoneType.MAINTENANCE_AREA:
            # Maintenance carts and monitoring stations
            zone_bounds = zone.bounds

            # 2 maintenance carts
            for cart_idx in range(2):
                cart_x = zone_bounds[0] + 3.0 + cart_idx * 4.0
                cart_y = zone_bounds[1] + 2.0

                cart = create_static_object(
                    name=f"maint_cart_{cart_idx}",
                    object_type="maintenance_cart",
                    position=(cart_x, cart_y, 0.5),
                    emits_sound=False,
                )
                static_objects.append(cart)

            # Monitoring station
            monitor_pos = ((zone_bounds[0] + zone_bounds[2])/2, zone_bounds[1] + 1.5, 0.6)
            monitor = create_static_object(
                name="noc_station",
                object_type="monitoring_station",
                position=monitor_pos,
                emits_sound=True,
                sound_type="equipment_buzz",
                sound_volume=0.2,
            )
            static_objects.append(monitor)

            if monitor.continuous_sound:
                continuous_sounds.append(monitor.continuous_sound)

    # Generate scenario-specific alerts
    alert_sounds = []

    if config.scenario_type == "patrol":
        # Normal patrol - occasional status beeps
        alert = create_preset_alert("beep", position=(width/2, depth/2, 1.5))
        alert_sounds.append((alert, 5.0))

    elif config.scenario_type == "fire_alarm":
        # Fire emergency - continuous alarm
        fire_alarm = create_preset_alert("alarm", position=(width * 0.3, depth * 0.2, 2.0))
        alert_sounds.append((fire_alarm, 1.0))
        # Evacuation warning
        evac_alert = create_preset_alert("warning", position=(width * 0.7, depth * 0.8, 2.0))
        alert_sounds.append((evac_alert, 3.0))

    elif config.scenario_type == "equipment_failure":
        # Server failure - error beeps
        error1 = create_preset_alert("error", position=(width * 0.25, depth * 0.4, 1.5))
        error2 = create_preset_alert("double_beep", position=(width * 0.25, depth * 0.4, 1.5))
        alert_sounds.append((error1, 0.5))
        alert_sounds.append((error2, 2.5))

    elif config.scenario_type == "maintenance":
        # Maintenance work - tool sounds and beeps
        beep = create_preset_alert("beep", position=(width * 0.8, depth * 0.9, 1.2))
        chime = create_preset_alert("chime", position=(width * 0.8, depth * 0.9, 1.2))
        alert_sounds.append((beep, 1.5))
        alert_sounds.append((chime, 4.0))

    # Compile metadata
    metadata = {
        "name": config.name,
        "floor_size": config.total_floor_size,
        "num_zones": len(zones),
        "total_racks": total_racks,
        "total_cooling_units": total_cooling,
        "total_power_units": total_power,
        "total_objects": len(static_objects),
        "total_continuous_sounds": len(continuous_sounds),
        "total_alerts": len(alert_sounds),
        "power_capacity_kw": total_racks * config.power_density,
        "zones": [
            {
                "type": z.zone_type.value,
                "bounds": z.bounds,
                "temperature": z.temperature,
                "restricted": z.access_restricted,
            }
            for z in zones
        ],
    }

    return static_objects, continuous_sounds, alert_sounds, metadata


# Convenience functions for common hyperscale scenarios

def scenario_hyperscale_small(seed=0):
    """Small hyperscale data center (500 racks, 2 zones).

    Suitable for initial robot training and testing.
    """
    config = HyperscaleConfig(
        name="Small_Hyperscale",
        total_floor_size=(60.0, 50.0),
        num_zones=2,
        racks_per_zone=250,
        layout_type="hot_aisle",
        include_support_rooms=True,
        scenario_type="patrol",
        seed=seed,
    )
    return generate_hyperscale_environment(config)


def scenario_hyperscale_medium(seed=0):
    """Medium hyperscale data center (1000 racks, 4 zones).

    Realistic commercial facility size.
    """
    config = HyperscaleConfig(
        name="Medium_Hyperscale",
        total_floor_size=(100.0, 80.0),
        num_zones=4,
        racks_per_zone=250,
        layout_type="hot_aisle",
        include_support_rooms=True,
        scenario_type="patrol",
        seed=seed,
    )
    return generate_hyperscale_environment(config)


def scenario_hyperscale_large(seed=0):
    """Large hyperscale data center (2000+ racks, 8 zones).

    Matches real Chinese hyperscale facilities (Alibaba, Tencent scale).
    """
    config = HyperscaleConfig(
        name="Large_Hyperscale_China",
        total_floor_size=(150.0, 120.0),
        num_zones=8,
        racks_per_zone=250,
        layout_type="hot_aisle",
        include_support_rooms=True,
        power_density=10.0,  # Higher density
        scenario_type="patrol",
        seed=seed,
    )
    return generate_hyperscale_environment(config)


def scenario_hyperscale_emergency(seed=0):
    """Hyperscale with fire alarm emergency.

    Robot must navigate evacuation during fire alarm.
    """
    config = HyperscaleConfig(
        name="Emergency_Scenario",
        total_floor_size=(100.0, 80.0),
        num_zones=4,
        racks_per_zone=250,
        scenario_type="fire_alarm",
        seed=seed,
    )
    return generate_hyperscale_environment(config)


# Example usage
if __name__ == "__main__":
    import json

    print("Hyperscale Data Center Generation")
    print("=" * 70)

    # Example 1: Small hyperscale
    print("\nExample 1: Small Hyperscale (500 racks)")
    objects, sounds, alerts, meta = scenario_hyperscale_small(seed=42)

    print(f"  Floor size: {meta['floor_size'][0]}m × {meta['floor_size'][1]}m")
    print(f"  Total racks: {meta['total_racks']}")
    print(f"  Cooling units: {meta['total_cooling_units']}")
    print(f"  Power capacity: {meta['power_capacity_kw']:.0f} kW")
    print(f"  Total objects: {meta['total_objects']}")
    print(f"  Continuous sounds: {meta['total_continuous_sounds']}")
    print(f"  Zones: {meta['num_zones']}")
    for zone_info in meta['zones']:
        print(f"    - {zone_info['type']}: {zone_info['bounds']}")

    # Example 2: Medium hyperscale
    print("\nExample 2: Medium Hyperscale (1000 racks)")
    objects2, sounds2, alerts2, meta2 = scenario_hyperscale_medium(seed=123)

    print(f"  Floor size: {meta2['floor_size'][0]}m × {meta2['floor_size'][1]}m")
    print(f"  Total racks: {meta2['total_racks']}")
    print(f"  Power capacity: {meta2['power_capacity_kw']:.0f} kW")
    print(f"  Total objects: {meta2['total_objects']}")

    # Example 3: Large hyperscale (China scale)
    print("\nExample 3: Large Hyperscale - China Scale (2000 racks)")
    objects3, sounds3, alerts3, meta3 = scenario_hyperscale_large(seed=999)

    print(f"  Name: {meta3['name']}")
    print(f"  Floor size: {meta3['floor_size'][0]}m × {meta3['floor_size'][1]}m")
    print(f"  Total racks: {meta3['total_racks']}")
    print(f"  Power capacity: {meta3['power_capacity_kw']:.0f} kW")
    print(f"  Total objects: {meta3['total_objects']}")
    print(f"  Continuous sounds: {meta3['total_continuous_sounds']}")

    # Example 4: Emergency scenario
    print("\nExample 4: Fire Emergency Scenario")
    objects4, sounds4, alerts4, meta4 = scenario_hyperscale_emergency(seed=555)

    print(f"  Scenario: {meta4['name']}")
    print(f"  Alert sounds: {meta4['total_alerts']}")
    for i, (alert, time) in enumerate(alerts4):
        print(f"    - {i+1}. {alert.alert_type} at {time}s")

    # Example 5: Save metadata
    print("\nExample 5: Saving metadata")
    with open("outputs/hyperscale_metadata.json", "w") as f:
        json.dump(meta3, f, indent=2)
    print("  Saved to: outputs/hyperscale_metadata.json")

    # Example 6: Generate MuJoCo XML
    print("\nExample 6: Generate MuJoCo XML (Medium Hyperscale)")
    from physonic.environments.objects import build_scene_with_static_objects

    xml = build_scene_with_static_objects(objects2, floor_size=meta2['floor_size'])
    with open("outputs/hyperscale_medium.xml", "w") as f:
        f.write(xml)
    print(f"  Saved to: outputs/hyperscale_medium.xml")
    print(f"  Contains {len(objects2)} objects across {meta2['num_zones']} zones")

    print("\nDone! Generated hyperscale data centers ready for commercial robot training.")
