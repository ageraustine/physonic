"""Room acoustics for spatial audio: early reflections + late reverb.

This module provides realistic room acoustics for indoor robot training environments.
Uses image-source method for early reflections and Schroeder reverberator for
late diffuse reverb.

Modules:
    - room: RoomSpec dataclass, image-source method, Schroeder reverberator
    - rooms: Predefined room templates (warehouse, data center, supermarket, etc.)

Example:
    >>> from physonic.acoustics import RoomSpec, apply_room_acoustics, ROOMS
    >>>
    >>> # Use predefined room
    >>> room = ROOMS['warehouse_large']
    >>>
    >>> # Or create custom room
    >>> room = RoomSpec(
    ...     dimensions=(30, 20, 8),
    ...     absorption=0.15,
    ...     rt60=2.5
    ... )
    >>>
    >>> # Apply room acoustics
    >>> import numpy as np
    >>> mono = np.random.randn(44100)  # 1 second audio
    >>> source_pos = np.array([5, 5, 1.5])
    >>> listener_pos = np.array([10, 10, 1.5])
    >>> reverb_audio = apply_room_acoustics(mono, source_pos, listener_pos, room)
"""
from physonic.acoustics.room import (
    RoomSpec,
    compute_image_sources,
    apply_image_source_reverb,
    SchroederReverberator,
    apply_room_acoustics
)
from physonic.acoustics.rooms import ROOMS, list_rooms

__all__ = [
    # Core types
    "RoomSpec",

    # Image-source method
    "compute_image_sources",
    "apply_image_source_reverb",

    # Late reverb
    "SchroederReverberator",

    # Main integration
    "apply_room_acoustics",

    # Room presets
    "ROOMS",
    "list_rooms",
]
