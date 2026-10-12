"""Predefined multi-object physics scenarios for robot acoustic training.

This module provides ready-to-use scenarios demonstrating complex contact
interactions: stacking, tumbling, cascading, scattering. Each scenario is
designed to produce distinctive acoustic signatures for robot perception tasks.
"""
from physonic.contact.rig import MultiSceneSpec, ObjectSpec


def scenario_stack_collapse(surface="wood", seed=0):
    """Three boxes stacked vertically, collapsing under gravity.

    Produces multiple distinct impact sounds as boxes hit each other and the ground.
    Useful for training robots to distinguish sequential vs simultaneous contacts.

    Args:
        surface: Surface material ("steel", "wood", "glass", "rubber")
        seed: Random seed for modal synthesis

    Returns:
        MultiSceneSpec with 3 boxes stacked
    """
    return MultiSceneSpec(
        kind="stack_collapse",
        surface=surface,
        objects=[
            ObjectSpec("box1", "box", "wood", 0.5, (0.05, 0.05, 0.02), (0, 0, 0.02)),
            ObjectSpec("box2", "box", "wood", 0.5, (0.05, 0.05, 0.02), (0, 0, 0.06)),
            ObjectSpec("box3", "box", "steel", 0.3, (0.04, 0.04, 0.02), (0, 0, 0.10)),
        ],
        seed=seed
    )


def scenario_sphere_cascade(surface="steel", n_spheres=5, seed=0):
    """Multiple spheres dropped in sequence, creating a cascade of impacts.

    Each sphere has a different material (alternating steel/glass) for timbral variety.
    Useful for training temporal pattern recognition in acoustic streams.

    Args:
        surface: Surface material
        n_spheres: Number of spheres (default: 5)
        seed: Random seed

    Returns:
        MultiSceneSpec with N spheres at different heights
    """
    objects = []
    for i in range(n_spheres):
        material = "steel" if i % 2 == 0 else "glass"
        # Stagger heights and slight horizontal offset
        height = 0.5 + i * 0.15
        x_offset = (i - n_spheres // 2) * 0.1

        objects.append(ObjectSpec(
            f"sphere{i}",
            "sphere",
            material,
            0.2,
            0.03,  # 3cm radius
            (x_offset, 0, height)
        ))

    return MultiSceneSpec(
        kind="sphere_cascade",
        surface=surface,
        objects=objects,
        seed=seed
    )


def scenario_tumble_box(surface="rubber", seed=0):
    """Single rectangular box tumbling with initial angular velocity.

    Produces multiple impacts from different faces hitting the ground.
    Useful for training robots to recognize rotational motion from acoustic cues.

    Args:
        surface: Surface material
        seed: Random seed

    Returns:
        MultiSceneSpec with 1 tumbling box
    """
    return MultiSceneSpec(
        kind="tumble",
        surface=surface,
        objects=[
            ObjectSpec(
                "box1",
                "box",
                "wood",
                0.5,
                (0.1, 0.05, 0.02),  # Rectangular: 10cm x 5cm x 2cm
                (0, 0, 0.3),
                initial_vel=(1.0, 0.5, 0)  # Horizontal velocity for tumbling
            )
        ],
        seed=seed
    )


def scenario_two_sphere_collision(surface="steel", seed=0):
    """Two spheres rolling toward each other, colliding mid-air before landing.

    Demonstrates object-object collision distinct from object-ground impacts.
    Critical for training robots to identify interaction types.

    Args:
        surface: Surface material
        seed: Random seed

    Returns:
        MultiSceneSpec with 2 spheres on collision course
    """
    return MultiSceneSpec(
        kind="collision",
        surface=surface,
        objects=[
            ObjectSpec(
                "sphere_left",
                "sphere",
                "steel",
                0.2,
                0.03,
                (-0.3, 0, 0.5),
                initial_vel=(2.0, 0, 0)  # Moving right
            ),
            ObjectSpec(
                "sphere_right",
                "sphere",
                "glass",
                0.2,
                0.03,
                (0.3, 0, 0.5),
                initial_vel=(-2.0, 0, 0)  # Moving left
            ),
        ],
        duration=1.5,  # Shorter: collision happens quickly
        seed=seed
    )


def scenario_scatter_pile(surface="wood", n_objects=6, seed=0):
    """Pile of mixed objects (spheres and boxes) scattering on impact.

    Creates chaotic multi-contact scenario with overlapping sounds.
    Useful for training source separation and multi-object tracking.

    Args:
        surface: Surface material
        n_objects: Number of objects in pile
        seed: Random seed

    Returns:
        MultiSceneSpec with N mixed objects in a cluster
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    objects = []
    for i in range(n_objects):
        shape = "sphere" if i % 2 == 0 else "box"
        material = rng.choice(["steel", "glass", "wood"])

        # Random position in a cluster
        x = rng.uniform(-0.15, 0.15)
        y = rng.uniform(-0.15, 0.15)
        z = 0.4 + i * 0.08  # Stacked vertically

        if shape == "sphere":
            objects.append(ObjectSpec(
                f"obj{i}",
                "sphere",
                material,
                0.15,
                0.025,  # 2.5cm radius
                (x, y, z)
            ))
        else:
            objects.append(ObjectSpec(
                f"obj{i}",
                "box",
                material,
                0.2,
                (0.03, 0.03, 0.02),
                (x, y, z)
            ))

    return MultiSceneSpec(
        kind="scatter",
        surface=surface,
        objects=objects,
        duration=3.0,  # Longer: objects scatter and settle
        seed=seed
    )


def scenario_domino_chain(surface="wood", n_dominoes=5, seed=0):
    """Chain of vertical boxes (dominoes) where first one is pushed over.

    Sequential impacts as each domino knocks over the next.
    Useful for training causal relationship detection from acoustic patterns.

    Args:
        surface: Surface material
        n_dominoes: Number of dominoes
        seed: Random seed

    Returns:
        MultiSceneSpec with N standing boxes in a row
    """
    objects = []
    spacing = 0.12  # 12cm between dominoes

    for i in range(n_dominoes):
        x_pos = -0.3 + i * spacing
        # Standing upright: tall in Z, thin in X/Y
        objects.append(ObjectSpec(
            f"domino{i}",
            "box",
            "wood",
            0.05,
            (0.01, 0.03, 0.06),  # 1cm x 3cm x 6cm (thin, standing)
            (x_pos, 0, 0.06)  # Placed at height = half-height
        ))

    # First domino gets a push
    objects[0] = ObjectSpec(
        "domino0",
        "box",
        "wood",
        0.05,
        (0.01, 0.03, 0.06),
        (-0.3, 0, 0.06),
        initial_vel=(0.3, 0, 0)  # Small push to start chain reaction
    )

    return MultiSceneSpec(
        kind="domino_chain",
        surface=surface,
        objects=objects,
        duration=3.0,
        seed=seed
    )


def scenario_rolling_cylinders(surface="steel", n_cylinders=3, seed=0):
    """Cylinders rolling down a slight incline and colliding.

    Demonstrates rolling friction sounds plus impact sounds.
    Useful for training robots to distinguish rolling vs sliding vs impact.

    Args:
        surface: Surface material
        n_cylinders: Number of cylinders
        seed: Random seed

    Returns:
        MultiSceneSpec with N cylinders on an incline
    """
    objects = []
    for i in range(n_cylinders):
        material = ["steel", "wood", "rubber"][i % 3]
        x_pos = -0.4 + i * 0.2
        z_pos = 0.3 + i * 0.1  # Offset vertically

        objects.append(ObjectSpec(
            f"cylinder{i}",
            "cylinder",
            material,
            0.3,
            (0.025, 0.05),  # radius 2.5cm, half-length 5cm
            (x_pos, 0, z_pos),
            initial_vel=(0.5, 0, 0)  # Initial rolling velocity
        ))

    return MultiSceneSpec(
        kind="rolling",
        surface=surface,
        objects=objects,
        duration=2.5,
        seed=seed
    )


# Scenario registry for CLI lookup
SCENARIOS = {
    "stack_collapse": scenario_stack_collapse,
    "sphere_cascade": scenario_sphere_cascade,
    "tumble_box": scenario_tumble_box,
    "two_sphere_collision": scenario_two_sphere_collision,
    "scatter_pile": scenario_scatter_pile,
    "domino_chain": scenario_domino_chain,
    "rolling_cylinders": scenario_rolling_cylinders,
}


def list_scenarios():
    """List all available scenarios with descriptions."""
    return {
        name: func.__doc__.split('\n\n')[0].strip()
        for name, func in SCENARIOS.items()
    }
