"""Static objects for MuJoCo environments.

This module defines static objects (server racks, pallets, shelves) that appear
in procedurally-generated environments. Static objects:
- Don't move under forces (mocap bodies)
- Participate in collision detection
- Can occlude sound sources (for spatial audio)
- Can emit continuous sounds (equipment)
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np

from physonic.synthesis.continuous import ContinuousSoundSpec


@dataclass
class StaticObjectSpec:
    """Static object in environment (doesn't move).

    Attributes:
        name: Unique object identifier (e.g., "server_rack_1")
        object_type: Type of object ("server_rack", "pallet_stack", "shelf_unit", etc.)
        position: (x, y, z) position in meters
        rotation: Yaw rotation in radians (rotation around Z axis)
        size: (width, depth, height) in meters
        material: Material name for visual/collision properties
        emits_continuous_sound: Whether object emits background sound
        continuous_sound: Sound specification (if emits_continuous_sound=True)
    """
    name: str
    object_type: str
    position: Tuple[float, float, float]
    rotation: float = 0.0  # radians
    size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    material: str = "steel"
    emits_continuous_sound: bool = False
    continuous_sound: Optional[ContinuousSoundSpec] = None


# Standard object dimensions (width, depth, height) in meters
OBJECT_SIZES = {
    # Data center objects - Compute & Storage
    "server_rack": (0.6, 1.0, 2.0),              # Standard 19" rack
    "storage_rack": (0.6, 1.2, 2.0),             # Deeper for disk arrays
    "network_rack": (0.6, 0.8, 2.0),             # Network switches
    "blade_server": (0.6, 0.8, 1.8),             # High-density compute

    # Data center objects - Power
    "pdu": (0.4, 0.6, 2.0),                      # Power Distribution Unit
    "ups_unit": (0.8, 1.2, 1.8),                 # Uninterruptible Power Supply
    "battery_cabinet": (0.6, 0.8, 2.0),          # Battery backup
    "transformer": (1.2, 1.5, 1.5),              # Power transformer

    # Data center objects - Cooling
    "cooling_unit": (0.8, 0.8, 1.5),             # Precision cooling (CRAC/CRAH)
    "crac_unit": (1.2, 1.2, 2.5),                # Large Computer Room Air Conditioner
    "hot_aisle_containment": (1.5, 0.1, 2.5),    # Containment door/panel
    "cold_aisle_containment": (1.5, 0.1, 2.5),   # Containment door/panel

    # Data center objects - Infrastructure
    "cable_tray": (0.3, 2.0, 0.1),               # Overhead cable management
    "cable_ladder": (0.5, 3.0, 0.15),            # Larger cable runs
    "raised_floor_panel": (0.6, 0.6, 0.4),       # Access floor
    "maintenance_cart": (0.8, 1.2, 1.0),         # Equipment cart
    "access_door": (1.2, 0.2, 2.4),              # Entry door
    "fire_suppression_panel": (0.6, 0.3, 1.5),   # Fire safety
    "monitoring_station": (1.5, 0.8, 1.2),       # NOC desk/station

    # Warehouse objects
    "pallet_stack_low": (1.2, 1.0, 1.0),
    "pallet_stack_medium": (1.2, 1.0, 1.5),
    "pallet_stack_high": (1.2, 1.0, 2.0),
    "shelving_unit": (2.0, 0.5, 3.0),
    "forklift": (1.5, 1.2, 2.0),

    # Supermarket objects
    "shelf_unit": (2.0, 0.5, 2.0),
    "shopping_cart": (1.0, 0.6, 1.0),
    "checkout_counter": (2.0, 0.6, 0.9),
    "refrigerator_unit": (2.5, 0.8, 2.2),
}

# Default materials for object types
OBJECT_MATERIALS = {
    # Data center - Compute & Storage
    "server_rack": "steel",
    "storage_rack": "steel",
    "network_rack": "steel",
    "blade_server": "steel",
    # Data center - Power
    "pdu": "steel",
    "ups_unit": "steel",
    "battery_cabinet": "steel",
    "transformer": "steel",
    # Data center - Cooling
    "cooling_unit": "steel",
    "crac_unit": "steel",
    "hot_aisle_containment": "glass",  # Transparent containment
    "cold_aisle_containment": "glass",
    # Data center - Infrastructure
    "cable_tray": "steel",
    "cable_ladder": "steel",
    "raised_floor_panel": "steel",
    "maintenance_cart": "steel",
    "access_door": "steel",
    "fire_suppression_panel": "steel",
    "monitoring_station": "wood",
    # Warehouse
    "pallet_stack_low": "wood",
    "pallet_stack_medium": "wood",
    "pallet_stack_high": "wood",
    "shelving_unit": "steel",
    "forklift": "steel",
    # Supermarket
    "shelf_unit": "steel",
    "shopping_cart": "steel",
    "checkout_counter": "wood",
    "refrigerator_unit": "steel",
}


def create_static_object(
    name: str,
    object_type: str,
    position: Tuple[float, float, float],
    rotation: float = 0.0,
    emits_sound: bool = False,
    sound_type: Optional[str] = None,
    sound_volume: float = 1.0,
) -> StaticObjectSpec:
    """Create a static object spec with standard dimensions.

    Args:
        name: Unique object identifier
        object_type: Type of object (must be in OBJECT_SIZES)
        position: (x, y, z) position in meters
        rotation: Yaw rotation in radians (default: 0)
        emits_sound: Whether object emits continuous sound
        sound_type: Type of sound ("server_hum", "fan", "cooling_system", etc.)
        sound_volume: Sound volume (0.0-1.0, default: 1.0)

    Returns:
        StaticObjectSpec with standard size and material

    Raises:
        ValueError: If object_type not in OBJECT_SIZES
    """
    if object_type not in OBJECT_SIZES:
        raise ValueError(
            f"Unknown object type '{object_type}'. "
            f"Available types: {list(OBJECT_SIZES.keys())}"
        )

    size = OBJECT_SIZES[object_type]
    material = OBJECT_MATERIALS[object_type]

    # Create continuous sound if requested
    continuous_sound = None
    if emits_sound and sound_type:
        from physonic.synthesis.continuous import create_preset_sound
        continuous_sound = create_preset_sound(
            source_id=name,
            sound_type=sound_type,
            position=position,
            volume=sound_volume,
        )

    return StaticObjectSpec(
        name=name,
        object_type=object_type,
        position=position,
        rotation=rotation,
        size=size,
        material=material,
        emits_continuous_sound=emits_sound,
        continuous_sound=continuous_sound,
    )


def build_static_object_xml(spec: StaticObjectSpec) -> str:
    """Generate MuJoCo XML for static object.

    Uses <body mocap="true"> to create non-dynamic object that:
    - Doesn't move under forces
    - Still participates in collision detection
    - Can occlude sound sources (for spatial audio)

    Args:
        spec: Static object specification

    Returns:
        XML fragment to insert in <worldbody>
    """
    x, y, z = spec.position
    width, depth, height = spec.size

    # Convert rotation from radians to quaternion (rotation around Z axis)
    # For rotation around Z: quat = [cos(θ/2), 0, 0, sin(θ/2)]
    cos_half = np.cos(spec.rotation / 2)
    sin_half = np.sin(spec.rotation / 2)
    quat = f"{cos_half:.6f} 0 0 {sin_half:.6f}"

    # Vibrant colors based on object type for better visibility
    if "rack" in spec.object_type:
        color = "0.2 0.4 0.8 1"  # Blue for server racks
    elif "cooling" in spec.object_type:
        color = "0.8 0.3 0.2 1"  # Red for cooling units
    elif "pallet" in spec.object_type:
        color = "0.6 0.4 0.2 1"  # Brown for pallets
    elif "shelf" in spec.object_type:
        color = "0.5 0.5 0.6 1"  # Gray for shelves
    else:
        color = "0.5 0.7 0.3 1"  # Green for other objects

    xml = f"""
    <!-- {spec.name} ({spec.object_type}) -->
    <body name="{spec.name}" pos="{x:.3f} {y:.3f} {z:.3f}" quat="{quat}" mocap="true">
      <geom type="box" size="{width/2:.3f} {depth/2:.3f} {height/2:.3f}"
            rgba="{color}" mass="1000" />
    </body>
"""
    return xml


def build_scene_with_static_objects(
    objects: List[StaticObjectSpec],
    floor_size: Tuple[float, float] = (50, 50),
) -> str:
    """Generate complete MuJoCo XML scene with static objects.

    Args:
        objects: List of static objects to include
        floor_size: (width, depth) of floor in meters

    Returns:
        Complete MuJoCo XML string
    """
    floor_width, floor_depth = floor_size

    # Build XML header with lighting and visual settings
    xml_parts = [
        '<?xml version="1.0" ?>',
        '<mujoco model="static_environment">',
        '  <option timestep="0.0002" gravity="0 0 -9.81" />',
        '',
        '  <!-- Visual settings for bright rendering -->',
        '  <visual>',
        '    <headlight ambient="0.8 0.8 0.8" diffuse="0.8 0.8 0.8" specular="0.5 0.5 0.5" />',
        '    <global offwidth="640" offheight="480" />',
        '  </visual>',
        '',
        '  <!-- Asset definitions -->',
        '  <asset>',
        '    <texture name="texplane" type="2d" builtin="checker" rgb1="0.4 0.4 0.5" rgb2="0.3 0.3 0.4" width="256" height="256" />',
        '    <material name="matplane" reflectance="0.2" texture="texplane" texrepeat="2 2" />',
        '  </asset>',
        '',
        '  <worldbody>',
        '    <!-- Bright directional lighting -->',
        '    <light directional="true" diffuse="1.2 1.2 1.2" specular="0.3 0.3 0.3" pos="0 0 5" dir="0 0 -1" castshadow="false" />',
        '    <light directional="true" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1" pos="10 10 5" dir="-1 -1 -1" castshadow="false" />',
        '',
        f'    <!-- Floor -->',
        f'    <geom type="plane" size="{floor_width} {floor_depth} 0.1" material="matplane" />',
        '',
    ]

    # Add static objects
    xml_parts.append('    <!-- Static Objects -->')
    for obj in objects:
        xml_parts.append(build_static_object_xml(obj))

    # Close worldbody and mujoco
    xml_parts.extend([
        '  </worldbody>',
        '</mujoco>',
    ])

    return '\n'.join(xml_parts)


# Example usage
if __name__ == "__main__":
    print("Static Objects Examples")
    print("=" * 60)

    # Example 1: Create individual objects
    print("\nExample 1: Creating static objects")

    rack1 = create_static_object(
        name="server_rack_1",
        object_type="server_rack",
        position=(2.0, 3.0, 1.0),
        rotation=0.0,
        emits_sound=True,
        sound_type="server_hum",
        sound_volume=0.7,
    )

    print(f"  Created: {rack1.name}")
    print(f"    Type: {rack1.object_type}")
    print(f"    Size: {rack1.size}")
    print(f"    Position: {rack1.position}")
    print(f"    Emits sound: {rack1.emits_continuous_sound}")
    if rack1.continuous_sound:
        print(f"    Sound type: {rack1.continuous_sound.sound_type}")

    # Example 2: Create multiple objects
    print("\nExample 2: Data center with multiple racks")

    objects = []
    for i in range(5):
        rack = create_static_object(
            name=f"server_rack_{i+1}",
            object_type="server_rack",
            position=(2.0 + i * 1.5, 3.0, 1.0),
            emits_sound=(i % 2 == 0),  # Every other rack emits sound
            sound_type="server_hum",
            sound_volume=0.6 + np.random.rand() * 0.2,
        )
        objects.append(rack)

    # Add cooling units
    for i in range(2):
        cooling = create_static_object(
            name=f"cooling_unit_{i+1}",
            object_type="cooling_unit",
            position=(2.0 + i * 8.0, 8.0, 0.75),
            emits_sound=True,
            sound_type="cooling_system",
            sound_volume=0.8,
        )
        objects.append(cooling)

    print(f"  Created {len(objects)} objects:")
    for obj in objects:
        sound_str = f"[{obj.continuous_sound.sound_type}]" if obj.emits_continuous_sound else ""
        print(f"    - {obj.name} at ({obj.position[0]:.1f}, {obj.position[1]:.1f}, {obj.position[2]:.1f}) {sound_str}")

    # Example 3: Generate MuJoCo XML
    print("\nExample 3: Generating MuJoCo XML")

    xml = build_scene_with_static_objects(objects, floor_size=(30, 20))

    # Save to file
    output_path = "outputs/static_objects_scene.xml"
    with open(output_path, "w") as f:
        f.write(xml)

    print(f"  Saved MuJoCo XML to: {output_path}")
    print(f"  Scene contains {len(objects)} static objects")

    # Example 4: Warehouse objects
    print("\nExample 4: Warehouse objects")

    warehouse_objects = [
        create_static_object(
            name="pallet_stack_1",
            object_type="pallet_stack_medium",
            position=(5.0, 5.0, 0.75),
        ),
        create_static_object(
            name="shelving_unit_1",
            object_type="shelving_unit",
            position=(10.0, 5.0, 1.5),
        ),
        create_static_object(
            name="forklift_1",
            object_type="forklift",
            position=(3.0, 10.0, 1.0),
            rotation=np.pi/4,  # 45 degrees
        ),
    ]

    print(f"  Created {len(warehouse_objects)} warehouse objects")

    # Example 5: All available object types
    print("\nExample 5: All available object types")
    print(f"  Total types: {len(OBJECT_SIZES)}")
    for obj_type, size in OBJECT_SIZES.items():
        material = OBJECT_MATERIALS[obj_type]
        print(f"    - {obj_type:25s} {size} ({material})")

    print("\nDone!")
