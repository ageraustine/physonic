"""Environment templates with procedural generation.

This module provides procedurally-generated commercial environment templates:
- Data centers (server racks, cooling units)
- Warehouses (shelving, pallets)
- Supermarkets (aisles, refrigerators)

Each environment includes:
- Procedural layout generation
- Static objects in MuJoCo scenes
- Background continuous sounds
- Dynamic events (impacts, alerts)
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
]
