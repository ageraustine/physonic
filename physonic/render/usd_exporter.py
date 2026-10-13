"""USD (Universal Scene Description) exporter for photorealistic rendering.

Exports MuJoCo scenes to USD format for use with:
- NVIDIA Isaac Sim (real-time photorealistic rendering)
- Blender (high-quality offline rendering)
- Other USD-compatible tools (Omniverse, Houdini, etc.)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import mujoco


@dataclass
class USDMaterial:
    """PBR (Physically-Based Rendering) material specification.

    Attributes:
        name: Material name
        base_color: RGB base color (0-1)
        metallic: Metallic factor (0=dielectric, 1=metal)
        roughness: Surface roughness (0=smooth, 1=rough)
        emissive: Emissive color for self-illumination (optional)
        normal_map: Path to normal map texture (optional)
        texture_path: Path to base color texture (optional)
    """
    name: str
    base_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: Optional[Tuple[float, float, float]] = None
    normal_map: Optional[str] = None
    texture_path: Optional[str] = None


# Photorealistic material presets for data centers
DATACENTER_MATERIALS = {
    "server_rack": USDMaterial(
        name="server_rack",
        base_color=(0.1, 0.1, 0.12),  # Dark metallic gray
        metallic=0.8,
        roughness=0.3,
    ),
    "server_front_panel": USDMaterial(
        name="server_front_panel",
        base_color=(0.05, 0.05, 0.05),  # Almost black
        metallic=0.9,
        roughness=0.2,
        emissive=(0.0, 0.3, 0.6),  # Blue LED glow
    ),
    "floor_raised": USDMaterial(
        name="floor_raised",
        base_color=(0.85, 0.85, 0.88),  # Light gray tiles
        metallic=0.0,
        roughness=0.6,
    ),
    "floor_concrete": USDMaterial(
        name="floor_concrete",
        base_color=(0.4, 0.4, 0.42),  # Concrete gray
        metallic=0.0,
        roughness=0.8,
    ),
    "cable_tray": USDMaterial(
        name="cable_tray",
        base_color=(0.7, 0.7, 0.72),  # Aluminum
        metallic=0.9,
        roughness=0.4,
    ),
    "cooling_unit": USDMaterial(
        name="cooling_unit",
        base_color=(0.9, 0.9, 0.92),  # Light metal
        metallic=0.7,
        roughness=0.3,
    ),
    "robot_body": USDMaterial(
        name="robot_body",
        base_color=(0.2, 0.25, 0.3),  # Dark blue-gray
        metallic=0.6,
        roughness=0.4,
    ),
    "led_green": USDMaterial(
        name="led_green",
        base_color=(0.0, 1.0, 0.0),
        metallic=0.0,
        roughness=0.1,
        emissive=(0.0, 2.0, 0.0),  # Bright green glow
    ),
    "led_red": USDMaterial(
        name="led_red",
        base_color=(1.0, 0.0, 0.0),
        metallic=0.0,
        roughness=0.1,
        emissive=(2.0, 0.0, 0.0),  # Bright red glow
    ),
}


@dataclass
class USDLight:
    """Light source specification for USD scene.

    Attributes:
        light_type: "point", "directional", "rect" (area light)
        position: Light position (x, y, z)
        color: RGB color (0-1)
        intensity: Light intensity
        radius: Light source radius (for soft shadows)
        direction: Light direction for directional/spot lights
    """
    light_type: str  # "point", "directional", "rect"
    position: Tuple[float, float, float]
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1000.0
    radius: float = 0.1
    direction: Optional[Tuple[float, float, float]] = None


class USDSceneExporter:
    """Export MuJoCo scenes to USD format with photorealistic materials.

    Supports:
    - PBR materials with metallic/roughness workflow
    - Multiple light sources (point, directional, area)
    - Instancing for repeated objects (server racks)
    - Camera definitions for rendering
    """

    def __init__(
        self,
        output_dir: str = "outputs/usd_scenes",
        use_instancing: bool = True,
    ):
        """Initialize USD exporter.

        Args:
            output_dir: Directory for USD output files
            use_instancing: Use instancing for repeated objects (more efficient)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_instancing = use_instancing

    def export_scene(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        scene_name: str,
        materials: Optional[Dict[str, USDMaterial]] = None,
        lights: Optional[List[USDLight]] = None,
        camera_position: Tuple[float, float, float] = (15, 15, 10),
        camera_target: Tuple[float, float, float] = (15, 12.5, 0),
    ) -> str:
        """Export MuJoCo scene to USD file.

        Args:
            model: MuJoCo model
            data: MuJoCo data (current state)
            scene_name: Output scene name
            materials: Material dictionary (defaults to DATACENTER_MATERIALS)
            lights: Light sources (defaults to data center lighting setup)
            camera_position: Camera position (x, y, z)
            camera_target: Camera look-at target

        Returns:
            Path to generated USD file
        """
        if materials is None:
            materials = DATACENTER_MATERIALS

        if lights is None:
            lights = self._create_datacenter_lighting(model)

        # Generate USD file path
        usd_path = self.output_dir / f"{scene_name}.usda"

        # Build USD scene (ASCII format for readability)
        usd_content = self._build_usd_scene(
            model, data, materials, lights,
            camera_position, camera_target
        )

        # Write USD file
        with open(usd_path, 'w') as f:
            f.write(usd_content)

        print(f"Exported USD scene to: {usd_path}")
        return str(usd_path)

    def _create_datacenter_lighting(self, model: mujoco.MjModel) -> List[USDLight]:
        """Create realistic data center lighting setup.

        Data centers typically have:
        - Overhead fluorescent/LED panels (cool white, ~5000K)
        - Even illumination (no harsh shadows)
        - Some areas with indicator LEDs

        Args:
            model: MuJoCo model for scene dimensions

        Returns:
            List of light sources
        """
        lights = []

        # Get room dimensions from model
        # Assuming floor geom is at index 0 (typical)
        room_width = 30.0  # Default, should extract from model
        room_depth = 25.0
        room_height = 4.0

        # Overhead area lights (simulate fluorescent panels)
        # Create grid of area lights for even illumination
        num_lights_x = 4
        num_lights_y = 3

        for i in range(num_lights_x):
            for j in range(num_lights_y):
                x = (i + 0.5) * room_width / num_lights_x
                y = (j + 0.5) * room_depth / num_lights_y
                z = room_height - 0.5  # Near ceiling

                lights.append(USDLight(
                    light_type="rect",
                    position=(x, y, z),
                    color=(1.0, 0.98, 0.95),  # Cool white
                    intensity=2000.0,
                    radius=1.0,  # Large for soft shadows
                    direction=(0, 0, -1),  # Point down
                ))

        # Ambient fill light (simulates indirect illumination)
        lights.append(USDLight(
            light_type="directional",
            position=(room_width / 2, room_depth / 2, room_height),
            color=(0.9, 0.92, 1.0),  # Slight blue tint
            intensity=500.0,
            direction=(0, 0, -1),
        ))

        return lights

    def _build_usd_scene(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        materials: Dict[str, USDMaterial],
        lights: List[USDLight],
        camera_position: Tuple[float, float, float],
        camera_target: Tuple[float, float, float],
    ) -> str:
        """Build USD scene content (ASCII format).

        Args:
            model: MuJoCo model
            data: MuJoCo data
            materials: Material dictionary
            lights: Light sources
            camera_position: Camera position
            camera_target: Camera target

        Returns:
            USD file content as string
        """
        # USD ASCII header
        usd_content = [
            '#usda 1.0',
            '(',
            '    defaultPrim = "World"',
            '    metersPerUnit = 1',
            '    upAxis = "Z"',
            ')',
            '',
            'def Xform "World"',
            '{',
        ]

        # Add materials
        usd_content.extend(self._export_materials(materials))

        # Add lights
        usd_content.extend(self._export_lights(lights))

        # Add camera
        usd_content.extend(self._export_camera(camera_position, camera_target))

        # Add geometry from MuJoCo model
        usd_content.extend(self._export_geometry(model, data, materials))

        # Close World xform
        usd_content.append('}')

        return '\n'.join(usd_content)

    def _export_materials(self, materials: Dict[str, USDMaterial]) -> List[str]:
        """Export PBR materials to USD format.

        Args:
            materials: Material dictionary

        Returns:
            USD material definitions
        """
        lines = ['    # Materials', '    def Scope "Materials"', '    {']

        for mat_name, mat in materials.items():
            lines.extend([
                f'        def Material "{mat_name}"',
                '        {',
                '            token outputs:surface.connect = </World/Materials/{}_Shader.outputs:surface>'.format(mat_name),
                '',
                f'            def Shader "{mat_name}_Shader"',
                '            {',
                '                uniform token info:id = "UsdPreviewSurface"',
                f'                color3f inputs:diffuseColor = {mat.base_color}',
                f'                float inputs:metallic = {mat.metallic}',
                f'                float inputs:roughness = {mat.roughness}',
            ])

            if mat.emissive:
                lines.append(f'                color3f inputs:emissiveColor = {mat.emissive}')

            lines.extend([
                '                token outputs:surface',
                '            }',
                '        }',
            ])

        lines.append('    }')
        lines.append('')
        return lines

    def _export_lights(self, lights: List[USDLight]) -> List[str]:
        """Export lights to USD format.

        Args:
            lights: Light source list

        Returns:
            USD light definitions
        """
        lines = ['    # Lights']

        for i, light in enumerate(lights):
            light_name = f"Light_{i}"
            light_type_map = {
                "point": "SphereLight",
                "directional": "DistantLight",
                "rect": "RectLight",
            }
            usd_light_type = light_type_map.get(light.light_type, "SphereLight")

            lines.extend([
                f'    def {usd_light_type} "{light_name}"',
                '    {',
                f'        float3 xformOp:translate = {light.position}',
                '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                f'        color3f inputs:color = {light.color}',
                f'        float inputs:intensity = {light.intensity}',
            ])

            if light.light_type == "rect":
                lines.append(f'        float inputs:width = {light.radius * 2}')
                lines.append(f'        float inputs:height = {light.radius * 2}')
            elif light.light_type == "point":
                lines.append(f'        float inputs:radius = {light.radius}')

            lines.append('    }')

        lines.append('')
        return lines

    def _export_camera(
        self,
        position: Tuple[float, float, float],
        target: Tuple[float, float, float],
    ) -> List[str]:
        """Export camera to USD format.

        Args:
            position: Camera position
            target: Camera look-at target

        Returns:
            USD camera definition
        """
        # Calculate camera orientation (look-at)
        # For simplicity, just set position; proper look-at requires rotation matrix

        lines = [
            '    # Camera',
            '    def Camera "MainCamera"',
            '    {',
            f'        float3 xformOp:translate = {position}',
            '        uniform token[] xformOpOrder = ["xformOp:translate"]',
            '        float focalLength = 50',
            '        float horizontalAperture = 36',
            '        float verticalAperture = 24',
            '    }',
            '',
        ]
        return lines

    def _export_geometry(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        materials: Dict[str, USDMaterial],
    ) -> List[str]:
        """Export MuJoCo geometry to USD format.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            materials: Material dictionary

        Returns:
            USD geometry definitions
        """
        lines = ['    # Geometry']

        # Export each geom as a USD primitive
        for geom_id in range(model.ngeom):
            geom_name = f"Geom_{geom_id}"
            geom_type = model.geom_type[geom_id]
            geom_size = model.geom_size[geom_id]
            geom_pos = data.geom_xpos[geom_id]

            # Determine USD primitive type
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                lines.extend(self._export_box(geom_name, geom_pos, geom_size, "floor_raised"))
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                lines.extend(self._export_sphere(geom_name, geom_pos, geom_size[0], "robot_body"))
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                lines.extend(self._export_cylinder(geom_name, geom_pos, geom_size, "server_rack"))
            # Add more geom types as needed

        return lines

    def _export_box(
        self,
        name: str,
        position: np.ndarray,
        size: np.ndarray,
        material_name: str,
    ) -> List[str]:
        """Export box geometry to USD.

        Args:
            name: Geometry name
            position: Position (x, y, z)
            size: Half-extents (x, y, z)
            material_name: Material to apply

        Returns:
            USD box definition
        """
        return [
            f'    def Cube "{name}"',
            '    {',
            f'        float3 xformOp:translate = ({position[0]}, {position[1]}, {position[2]})',
            f'        float3 xformOp:scale = ({size[0]}, {size[1]}, {size[2]})',
            '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
            f'        rel material:binding = </World/Materials/{material_name}>',
            '    }',
        ]

    def _export_sphere(
        self,
        name: str,
        position: np.ndarray,
        radius: float,
        material_name: str,
    ) -> List[str]:
        """Export sphere geometry to USD.

        Args:
            name: Geometry name
            position: Position (x, y, z)
            radius: Sphere radius
            material_name: Material to apply

        Returns:
            USD sphere definition
        """
        return [
            f'    def Sphere "{name}"',
            '    {',
            f'        float3 xformOp:translate = ({position[0]}, {position[1]}, {position[2]})',
            f'        float3 xformOp:scale = ({radius}, {radius}, {radius})',
            '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
            f'        rel material:binding = </World/Materials/{material_name}>',
            '    }',
        ]

    def _export_cylinder(
        self,
        name: str,
        position: np.ndarray,
        size: np.ndarray,
        material_name: str,
    ) -> List[str]:
        """Export cylinder geometry to USD.

        Args:
            name: Geometry name
            position: Position (x, y, z)
            size: [radius, half-height]
            material_name: Material to apply

        Returns:
            USD cylinder definition
        """
        radius = size[0]
        height = size[1] * 2

        return [
            f'    def Cylinder "{name}"',
            '    {',
            f'        float3 xformOp:translate = ({position[0]}, {position[1]}, {position[2]})',
            f'        float3 xformOp:scale = ({radius}, {radius}, {height})',
            '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
            f'        rel material:binding = </World/Materials/{material_name}>',
            '    }',
        ]
