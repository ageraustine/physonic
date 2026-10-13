"""Environment templates with procedural generation.

This module provides procedurally-generated commercial environment templates:
- Data centers (basic: 10-24 racks, hyperscale: 500-2000+ racks)
- Warehouses (shelving, pallets) - coming soon
- Supermarkets (aisles, refrigerators) - coming soon

Each environment includes:
- Procedural layout generation (grid-based collision detection)
- Static objects in MuJoCo scenes (34 object types)
- Background continuous sounds (equipment, HVAC, cooling)
- Dynamic events (impacts, alerts, alarms)

Hyperscale data centers:
- Multi-zone architecture (server halls, UPS rooms, cooling plants)
- 100s to 1000s of server racks
- Realistic infrastructure (power distribution, cooling, cable management)
- Zone-specific acoustic signatures
- Advanced robot training scenarios (patrol, emergency response, maintenance)
"""

from physonic.environments.layout import (
    LayoutGrid,
    LayoutConstraints,
)
from physonic.environments.objects import (
    StaticObjectSpec,
    create_static_object,
    build_static_object_xml,
    build_scene_with_static_objects,
    OBJECT_SIZES,
    OBJECT_MATERIALS,
)
from physonic.environments.data_center_hyperscale import (
    HyperscaleConfig,
    ZoneType,
    RackType,
    Zone,
    generate_hyperscale_environment,
    scenario_hyperscale_small,
    scenario_hyperscale_medium,
    scenario_hyperscale_large,
    scenario_hyperscale_emergency,
)
from physonic.environments.scene_optimizer import (
    CompoundObjectGroup,
    build_optimized_static_object_xml,
    build_compound_body_xml,
    create_spatial_groups,
    build_optimized_scene_xml,
)

__all__ = [
    # Layout
    "LayoutGrid",
    "LayoutConstraints",
    # Static objects
    "StaticObjectSpec",
    "create_static_object",
    "build_static_object_xml",
    "build_scene_with_static_objects",
    "OBJECT_SIZES",
    "OBJECT_MATERIALS",
    # Scene optimization (MuJoCo/Gymnasium/Isaac Sim best practices)
    "CompoundObjectGroup",
    "build_optimized_static_object_xml",
    "build_compound_body_xml",
    "create_spatial_groups",
    "build_optimized_scene_xml",
    # Hyperscale data centers
    "HyperscaleConfig",
    "ZoneType",
    "RackType",
    "Zone",
    "generate_hyperscale_environment",
    "scenario_hyperscale_small",
    "scenario_hyperscale_medium",
    "scenario_hyperscale_large",
    "scenario_hyperscale_emergency",
]
