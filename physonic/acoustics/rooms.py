"""Predefined room acoustic environments for robot training scenarios.

This module provides realistic room presets for common indoor robot environments:
warehouses, data centers, supermarkets, offices, and hallways. Each preset is
tuned with appropriate dimensions, absorption coefficients, and RT60 values based
on typical architectural acoustics.
"""
from typing import Dict

from physonic.acoustics.room import RoomSpec

# =============================================================================
# Warehouse Environments
# =============================================================================

WAREHOUSE_SMALL = RoomSpec(
    dimensions=(15, 10, 5),      # 15m × 10m × 5m
    absorption=0.2,              # Concrete walls, some clutter/boxes
    rt60=1.5,                    # Moderate reverb
    max_reflection_order=2
)

WAREHOUSE_LARGE = RoomSpec(
    dimensions=(30, 20, 8),      # 30m × 20m × 8m tall
    absorption=0.15,             # Hard concrete/metal surfaces
    rt60=2.5,                    # Long reverb (large, reflective space)
    max_reflection_order=2
)

WAREHOUSE_EMPTY = RoomSpec(
    dimensions=(40, 30, 10),     # Very large, empty warehouse
    absorption=0.1,              # Minimal absorption (bare walls)
    rt60=3.5,                    # Very long reverb
    max_reflection_order=2
)

# =============================================================================
# Data Center Environments
# =============================================================================

DATA_CENTER_STANDARD = RoomSpec(
    dimensions=(20, 15, 3),      # Typical data center floor
    absorption=0.4,              # Server racks absorb significant sound
    rt60=0.6,                    # Short reverb (lots of equipment)
    max_reflection_order=2
)

DATA_CENTER_LARGE = RoomSpec(
    dimensions=(40, 30, 3),      # Large data center
    absorption=0.35,             # Dense server racks
    rt60=0.8,                    # Short reverb
    max_reflection_order=2
)

# =============================================================================
# Supermarket Environments
# =============================================================================

SUPERMARKET_SMALL = RoomSpec(
    dimensions=(20, 15, 4),      # Small grocery store
    absorption=0.3,              # Shelving, products, some soft goods
    rt60=0.9,                    # Moderate reverb
    max_reflection_order=2
)

SUPERMARKET_MEDIUM = RoomSpec(
    dimensions=(40, 25, 4),      # Medium supermarket
    absorption=0.25,             # Mix of hard (floors) and soft (goods) surfaces
    rt60=1.2,                    # Moderate reverb
    max_reflection_order=2
)

SUPERMARKET_LARGE = RoomSpec(
    dimensions=(60, 40, 5),      # Large supermarket
    absorption=0.22,             # Mostly hard surfaces with some absorption
    rt60=1.5,                    # Longer reverb (large space)
    max_reflection_order=2
)

# =============================================================================
# Office Environments
# =============================================================================

OFFICE_SMALL = RoomSpec(
    dimensions=(6, 4, 2.7),      # Small office room
    absorption=0.5,              # Carpets, furniture, acoustic panels
    rt60=0.4,                    # Short reverb (well-damped)
    max_reflection_order=3       # Higher order for small spaces
)

OFFICE_MEDIUM = RoomSpec(
    dimensions=(12, 8, 2.7),     # Medium office or conference room
    absorption=0.4,              # Mix of hard and soft surfaces
    rt60=0.6,                    # Moderate reverb
    max_reflection_order=2
)

OFFICE_OPEN_PLAN = RoomSpec(
    dimensions=(30, 20, 2.7),    # Open-plan office
    absorption=0.35,             # Some acoustic treatment
    rt60=0.8,                    # Moderate reverb
    max_reflection_order=2
)

# =============================================================================
# Hallway/Corridor Environments
# =============================================================================

HALLWAY_SHORT = RoomSpec(
    dimensions=(10, 2, 2.7),     # Short narrow hallway
    absorption=0.35,             # Drywall, some acoustic treatment
    rt60=0.6,                    # Flutter echoes possible
    max_reflection_order=2
)

HALLWAY_LONG = RoomSpec(
    dimensions=(20, 2, 2.7),     # Long narrow hallway
    absorption=0.3,              # Hard surfaces (tiles, painted walls)
    rt60=0.8,                    # Noticeable reverb/flutter
    max_reflection_order=2
)

# =============================================================================
# Industrial Environments
# =============================================================================

FACTORY_FLOOR = RoomSpec(
    dimensions=(50, 40, 8),      # Factory production floor
    absorption=0.2,              # Hard concrete, metal equipment
    rt60=2.0,                    # Long reverb (large, hard space)
    max_reflection_order=2
)

LOADING_DOCK = RoomSpec(
    dimensions=(15, 12, 6),      # Loading dock area
    absorption=0.25,             # Mixed surfaces
    rt60=1.3,                    # Moderate-long reverb
    max_reflection_order=2
)

# =============================================================================
# Special Environments
# =============================================================================

ANECHOIC = RoomSpec(
    dimensions=(10, 8, 3),       # Anechoic chamber (no reverb)
    absorption=0.99,             # Nearly complete absorption
    rt60=0.1,                    # Minimal reverb
    max_reflection_order=1       # Only first-order reflections matter
)

REVERBERANT_CHAMBER = RoomSpec(
    dimensions=(8, 6, 4),        # Reverberant test chamber
    absorption=0.05,             # Highly reflective walls
    rt60=4.0,                    # Very long decay
    max_reflection_order=3       # Higher-order reflections significant
)

# =============================================================================
# Room Registry for CLI Lookup
# =============================================================================

ROOMS = {
    # Warehouses
    "warehouse_small": WAREHOUSE_SMALL,
    "warehouse_large": WAREHOUSE_LARGE,
    "warehouse_empty": WAREHOUSE_EMPTY,

    # Data centers
    "data_center": DATA_CENTER_STANDARD,
    "data_center_large": DATA_CENTER_LARGE,

    # Supermarkets
    "supermarket_small": SUPERMARKET_SMALL,
    "supermarket": SUPERMARKET_MEDIUM,
    "supermarket_large": SUPERMARKET_LARGE,

    # Offices
    "office_small": OFFICE_SMALL,
    "office": OFFICE_MEDIUM,
    "office_open": OFFICE_OPEN_PLAN,

    # Hallways
    "hallway_short": HALLWAY_SHORT,
    "hallway": HALLWAY_LONG,

    # Industrial
    "factory": FACTORY_FLOOR,
    "loading_dock": LOADING_DOCK,

    # Special
    "anechoic": ANECHOIC,
    "reverb_chamber": REVERBERANT_CHAMBER,
}


def list_rooms() -> Dict[str, str]:
    """List all available room presets with brief descriptions.

    Returns:
        Dictionary mapping room name to description string
    """
    descriptions = {
        "warehouse_small": "Small warehouse (15×10×5m, RT60=1.5s)",
        "warehouse_large": "Large warehouse (30×20×8m, RT60=2.5s)",
        "warehouse_empty": "Empty warehouse (40×30×10m, RT60=3.5s)",
        "data_center": "Standard data center (20×15×3m, RT60=0.6s)",
        "data_center_large": "Large data center (40×30×3m, RT60=0.8s)",
        "supermarket_small": "Small grocery store (20×15×4m, RT60=0.9s)",
        "supermarket": "Medium supermarket (40×25×4m, RT60=1.2s)",
        "supermarket_large": "Large supermarket (60×40×5m, RT60=1.5s)",
        "office_small": "Small office (6×4×2.7m, RT60=0.4s)",
        "office": "Medium office (12×8×2.7m, RT60=0.6s)",
        "office_open": "Open-plan office (30×20×2.7m, RT60=0.8s)",
        "hallway_short": "Short hallway (10×2×2.7m, RT60=0.6s)",
        "hallway": "Long hallway (20×2×2.7m, RT60=0.8s)",
        "factory": "Factory floor (50×40×8m, RT60=2.0s)",
        "loading_dock": "Loading dock (15×12×6m, RT60=1.3s)",
        "anechoic": "Anechoic chamber (10×8×3m, RT60=0.1s)",
        "reverb_chamber": "Reverb chamber (8×6×4m, RT60=4.0s)",
    }
    return descriptions
