"""Contact sound synthesis from physics simulation (mono audio).

This module handles:
- Physics simulation with MuJoCo (contact forces, velocities)
- Event extraction (impacts from force traces)
- Modal audio synthesis (resonant modes, friction)
- Multi-object scenarios (collisions, stacking, tumbling)

Modules:
    - rig: MuJoCo scene builder and physics simulation
    - modal_synth: Modal audio synthesis (impacts, friction)
    - materials: Material properties (physics + acoustics)
"""
from physonic.contact.materials import MATERIALS, Material
from physonic.contact.rig import (
    SceneSpec, simulate, extract_events, OBJ_MASS, OBJ_SIZE,
    ObjectSpec, MultiSceneSpec, build_multi_xml, simulate_multi, extract_events_multi
)
from physonic.contact import modal_synth

__all__ = [
    "MATERIALS",
    "Material",
    "SceneSpec",
    "simulate",
    "extract_events",
    "OBJ_MASS",
    "OBJ_SIZE",
    "modal_synth",
    # Multi-object APIs
    "ObjectSpec",
    "MultiSceneSpec",
    "build_multi_xml",
    "simulate_multi",
    "extract_events_multi",
]
