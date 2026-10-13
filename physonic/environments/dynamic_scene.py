"""Dynamic data center scene with physics and moving sounds.

This module creates complex data center scenarios with:
- Dynamic objects (not static mocap)
- Moving robots/equipment
- Spatial audio tracking for moving sources
- Physics simulation for realistic interactions
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

from physonic.environments.objects import StaticObjectSpec, build_static_object_xml
from physonic.synthesis.continuous import ContinuousSoundSpec
from physonic.synthesis.alerts import AlertSoundSpec


@dataclass
class DynamicObjectSpec:
    """Specification for dynamic object with physics.

    Unlike StaticObjectSpec (mocap=true), these objects have:
    - Mass and inertia
    - Can move under forces
    - Collision dynamics
    """
    name: str
    object_type: str
    position: Tuple[float, float, float]
    rotation: float  # Radians around Z axis
    size: Tuple[float, float, float]  # (width, depth, height)
    material: str = "steel"
    mass: float = 100.0  # kg

    # Movement (for robots, carts)
    is_mobile: bool = False
    velocity: Optional[Tuple[float, float, float]] = None  # (vx, vy, vz) m/s

    # Sound emission
    emits_continuous_sound: bool = False
    continuous_sound: Optional[ContinuousSoundSpec] = None
    sound_follows_object: bool = False  # If True, sound position updates with object


@dataclass
class MovingRobotSpec:
    """Specification for a robot that moves through the data center.

    The robot:
    - Moves along a path
    - Emits motor sounds
    - Position updates over time
    """
    name: str
    start_position: Tuple[float, float, float]
    end_position: Tuple[float, float, float]
    speed: float  # m/s
    size: Tuple[float, float, float] = (1.2, 1.2, 1.2)  # Much larger for visibility
    mass: float = 120.0

    # Sound
    motor_sound_freq: float = 80.0  # Hz
    motor_sound_volume: float = 0.5


def build_dynamic_object_xml(spec: DynamicObjectSpec) -> str:
    """Generate MuJoCo XML for dynamic object with physics.

    Args:
        spec: Dynamic object specification

    Returns:
        XML fragment with body, geom, joint, and inertial properties
    """
    x, y, z = spec.position
    width, depth, height = spec.size

    # Convert rotation to quaternion
    cos_half = np.cos(spec.rotation / 2)
    sin_half = np.sin(spec.rotation / 2)
    quat = f"{cos_half:.6f} 0 0 {sin_half:.6f}"

    # Color based on type
    if "rack" in spec.object_type:
        color = "0.2 0.4 0.8 1"  # Blue
    elif "cooling" in spec.object_type:
        color = "0.8 0.3 0.2 1"  # Red
    elif "robot" in spec.object_type:
        color = "0.3 0.8 0.3 1"  # Green
    elif "cart" in spec.object_type:
        color = "0.8 0.7 0.2 1"  # Yellow
    else:
        color = "0.5 0.7 0.3 1"

    # Calculate inertia (box inertia tensor)
    # I = m/12 * (h^2 + d^2, w^2 + h^2, w^2 + d^2)
    ixx = spec.mass / 12 * (height**2 + depth**2)
    iyy = spec.mass / 12 * (width**2 + height**2)
    izz = spec.mass / 12 * (width**2 + depth**2)

    xml = f"""
    <!-- {spec.name} ({spec.object_type}) - DYNAMIC -->
    <body name="{spec.name}" pos="{x:.3f} {y:.3f} {z:.3f}" quat="{quat}">
      <inertial pos="0 0 0" mass="{spec.mass:.2f}" diaginertia="{ixx:.3f} {iyy:.3f} {izz:.3f}" />
      <geom type="box" size="{width/2:.3f} {depth/2:.3f} {height/2:.3f}"
            rgba="{color}" friction="0.8 0.005 0.0001" />
      <freejoint/>
    </body>
"""
    return xml


def build_moving_robot_xml(spec: MovingRobotSpec) -> str:
    """Generate MuJoCo XML for moving robot.

    Args:
        spec: Moving robot specification

    Returns:
        XML fragment for robot with actuator
    """
    x, y, z = spec.start_position
    width, depth, height = spec.size

    # Inertia
    ixx = spec.mass / 12 * (height**2 + depth**2)
    iyy = spec.mass / 12 * (width**2 + height**2)
    izz = spec.mass / 12 * (width**2 + depth**2)

    # Robot with bright marker on top for visibility
    xml = f"""
    <!-- {spec.name} - MOVING ROBOT (Bright Orange with Red Marker) -->
    <body name="{spec.name}" pos="{x:.3f} {y:.3f} {z:.3f}">
      <inertial pos="0 0 0" mass="{spec.mass:.2f}" diaginertia="{ixx:.3f} {iyy:.3f} {izz:.3f}" />
      <!-- Main robot body (sphere) -->
      <geom type="sphere" size="{width/2:.3f}"
            rgba="1.0 0.5 0.0 1" friction="0.5 0.005 0.0001" />
      <!-- Bright marker on top -->
      <geom type="cylinder" size="0.15 0.3" pos="0 0 {height/2 + 0.3:.3f}"
            rgba="1.0 0.0 0.0 1" />
      <joint name="{spec.name}_slide_x" type="slide" axis="1 0 0" limited="false" damping="10" />
      <joint name="{spec.name}_slide_y" type="slide" axis="0 1 0" limited="false" damping="10" />
      <joint name="{spec.name}_hinge" type="hinge" axis="0 0 1" limited="false" damping="1" />
    </body>
"""
    return xml


def build_complex_scene_xml(
    static_objects: List[StaticObjectSpec],
    moving_robots: List[MovingRobotSpec],
    floor_size: Tuple[float, float] = (30, 25),
) -> str:
    """Build complete MuJoCo XML for modern data center scene with static equipment.

    Creates a realistic data center where:
    - Server racks are static (bolted to raised floor, can't fall)
    - Cooling units are mounted
    - Robots can move freely
    - Professional lighting and materials

    Args:
        static_objects: List of static objects (racks, cooling units)
        moving_robots: List of moving robots
        floor_size: (width, depth) of floor

    Returns:
        Complete MuJoCo XML string
    """
    floor_width, floor_depth = floor_size

    xml_parts = [
        '<?xml version="1.0" ?>',
        '<mujoco model="complex_data_center">',
        '  <option timestep="0.001" gravity="0 0 -9.81" />',
        '',
        '  <!-- Visual settings -->',
        '  <visual>',
        '    <headlight ambient="0.8 0.8 0.8" diffuse="0.8 0.8 0.8" specular="0.5 0.5 0.5" />',
        '    <global offwidth="640" offheight="480" />',
        '  </visual>',
        '',
        '  <!-- Asset definitions -->',
        '  <asset>',
        '    <!-- Raised floor tiles (anti-static, light gray) -->',
        '    <texture name="texplane" type="2d" builtin="checker" rgb1="0.75 0.75 0.78" rgb2="0.7 0.7 0.73" width="512" height="512" />',
        '    <material name="matplane" reflectance="0.4" texture="texplane" texrepeat="15 15" />',
        '  </asset>',
        '',
        '  <worldbody>',
        '    <!-- Modern data center lighting (bright, even illumination) -->',
        '    <light directional="true" diffuse="1.4 1.4 1.45" specular="0.4 0.4 0.4" pos="0 0 8" dir="0 0 -1" castshadow="false" />',
        '    <light directional="true" diffuse="0.8 0.8 0.82" specular="0.2 0.2 0.2" pos="15 12.5 6" dir="0 0 -1" castshadow="false" />',
        '    <light directional="true" diffuse="0.5 0.5 0.52" specular="0.1 0.1 0.1" pos="-5 20 5" dir="1 -1 -1" castshadow="false" />',
        '',
        f'    <!-- Floor -->',
        f'    <geom type="plane" size="{floor_width} {floor_depth} 0.1" material="matplane" />',
        '',
        '    <!-- Walls for containment -->',
        f'    <geom type="box" pos="{floor_width} {floor_depth/2} 1.5" size="0.1 {floor_depth/2} 1.5" rgba="0.6 0.6 0.7 0.3" />',
        f'    <geom type="box" pos="0 {floor_depth/2} 1.5" size="0.1 {floor_depth/2} 1.5" rgba="0.6 0.6 0.7 0.3" />',
        f'    <geom type="box" pos="{floor_width/2} {floor_depth} 1.5" size="{floor_width/2} 0.1 1.5" rgba="0.6 0.6 0.7 0.3" />',
        f'    <geom type="box" pos="{floor_width/2} 0 1.5" size="{floor_width/2} 0.1 1.5" rgba="0.6 0.6 0.7 0.3" />',
        '',
    ]

    # Add static objects (server racks, cooling units)
    xml_parts.append('    <!-- Static Objects (Bolted Equipment) -->')
    for obj in static_objects:
        xml_parts.append(build_static_object_xml(obj))

    # Add moving robots
    xml_parts.append('    <!-- Moving Robots -->')
    for robot in moving_robots:
        xml_parts.append(build_moving_robot_xml(robot))

    # Add actuators for robots
    if moving_robots:
        xml_parts.append('  </worldbody>')
        xml_parts.append('')
        xml_parts.append('  <!-- Actuators for robot movement -->')
        xml_parts.append('  <actuator>')
        for robot in moving_robots:
            xml_parts.append(f'    <motor joint="{robot.name}_slide_x" gear="100" ctrllimited="true" ctrlrange="-10 10" />')
            xml_parts.append(f'    <motor joint="{robot.name}_slide_y" gear="100" ctrllimited="true" ctrlrange="-10 10" />')
            xml_parts.append(f'    <motor joint="{robot.name}_hinge" gear="20" ctrllimited="true" ctrlrange="-5 5" />')
        xml_parts.append('  </actuator>')
    else:
        xml_parts.append('  </worldbody>')

    xml_parts.append('</mujoco>')

    return '\n'.join(xml_parts)


def generate_complex_data_center(
    room_size: Tuple[float, float, float] = (30, 25, 4),
    num_racks: int = 24,
    num_robots: int = 2,
    seed: int = 0,
) -> Tuple[List[StaticObjectSpec], List[MovingRobotSpec], List[ContinuousSoundSpec]]:
    """Generate modern data center with static racks and moving robots.

    Creates a realistic data center where:
    - Server racks are bolted to raised floor (static, can't fall)
    - Cooling units are mounted (static)
    - Robots can move freely
    - Equipment emits realistic sounds

    Args:
        room_size: (width, depth, height) in meters
        num_racks: Number of server racks
        num_robots: Number of moving robots
        seed: Random seed

    Returns:
        Tuple of (static_objects, moving_robots, continuous_sounds)
    """
    rng = np.random.default_rng(seed)
    room_width, room_depth, room_height = room_size

    static_objects = []  # Changed from dynamic_objects
    moving_robots = []
    continuous_sounds = []

    # Generate server racks in hot-aisle/cold-aisle layout
    # Modern data centers use alternating aisles for airflow efficiency
    racks_per_row = 6
    num_rows = (num_racks + racks_per_row - 1) // racks_per_row

    rack_x_start = 3.0
    rack_y_start = 3.0
    rack_spacing_x = 2.2  # Standard 42U rack spacing
    rack_spacing_y = 3.5  # Aisle width: 1.5m between rows

    for i in range(num_racks):
        row = i // racks_per_row
        col = i % racks_per_row

        x = rack_x_start + col * rack_spacing_x
        y = rack_y_start + row * rack_spacing_y
        z = 1.0  # Racks sit on raised floor (no random height in real data centers)

        emits_sound = (i % 3 != 0)  # 2/3 of racks emit sound (active servers)

        # Use StaticObjectSpec - racks are bolted to floor, can't fall over
        rack = StaticObjectSpec(
            name=f"rack_{i+1}",
            object_type="server_rack",
            position=(x, y, z),
            rotation=0.0,  # All racks face same direction
            size=(0.6, 1.0, 2.0),  # Standard 42U rack: 60cm wide, 100cm deep, 2m tall
            material="steel",
            emits_continuous_sound=emits_sound,
        )

        static_objects.append(rack)

        if emits_sound:
            sound = ContinuousSoundSpec(
                source_id=f"rack_{i+1}",
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

    # Add cooling units (CRAC - Computer Room Air Conditioning)
    # Placed around perimeter in modern data centers
    cooling_positions = [
        (5, room_depth - 2, 0.75),   # Back wall
        (15, room_depth - 2, 0.75),  # Back wall
        (25, room_depth - 2, 0.75),  # Back wall
        (room_width - 2, 10, 0.75),  # Side wall
    ]

    for i, pos in enumerate(cooling_positions):
        # Use StaticObjectSpec - cooling units are mounted, can't fall
        cooling = StaticObjectSpec(
            name=f"cooling_{i+1}",
            object_type="cooling_unit",
            position=pos,
            rotation=0.0,
            size=(0.8, 0.8, 1.5),  # CRAC unit size
            material="steel",
            emits_continuous_sound=True,
        )
        static_objects.append(cooling)

        sound = ContinuousSoundSpec(
            source_id=f"cooling_{i+1}",
            sound_type="cooling_system",
            fundamental_freq=180.0,
            harmonics=[1.0, 0.6, 0.4],
            noise_amount=0.4,
            modulation_freq=3.0,
            modulation_depth=0.25,
            position=pos,
            volume=0.7,
        )
        continuous_sounds.append(sound)

    # Add moving robots
    for i in range(num_robots):
        # Random patrol path
        start_x = rng.uniform(2, room_width - 2)
        start_y = rng.uniform(2, room_depth - 2)
        end_x = rng.uniform(2, room_width - 2)
        end_y = rng.uniform(2, room_depth - 2)

        robot = MovingRobotSpec(
            name=f"robot_{i+1}",
            start_position=(start_x, start_y, 0.6),  # Higher for larger robot
            end_position=(end_x, end_y, 0.6),
            speed=0.5,  # 0.5 m/s
            mass=120.0,  # Match default mass
        )
        moving_robots.append(robot)

        # Robot motor sound (will move with robot)
        sound = ContinuousSoundSpec(
            source_id=f"robot_{i+1}",
            sound_type="fan",  # Motor sound
            fundamental_freq=robot.motor_sound_freq,
            harmonics=[1.0, 0.7, 0.4, 0.2],
            noise_amount=0.3,
            modulation_freq=4.0,
            modulation_depth=0.3,
            position=robot.start_position,
            volume=robot.motor_sound_volume,
        )
        continuous_sounds.append(sound)

    return static_objects, moving_robots, continuous_sounds


# Example usage
if __name__ == "__main__":
    print("Complex Data Center Generation")
    print("=" * 60)

    dynamic_objects, moving_robots, continuous_sounds = generate_complex_data_center(
        room_size=(30, 25, 4),
        num_racks=24,
        num_robots=2,
        seed=42,
    )

    print(f"\nGenerated:")
    print(f"  Dynamic objects: {len(dynamic_objects)}")
    print(f"  Moving robots: {len(moving_robots)}")
    print(f"  Continuous sounds: {len(continuous_sounds)}")

    # Build XML
    xml = build_complex_scene_xml(dynamic_objects, moving_robots, floor_size=(30, 25))

    with open("outputs/complex_data_center.xml", "w") as f:
        f.write(xml)

    print(f"\nSaved to: outputs/complex_data_center.xml")
