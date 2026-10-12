"""MuJoCo scene builder + simulation with contact telemetry logging."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import mujoco
import numpy as np
from scipy.ndimage import uniform_filter1d

from .materials import MATERIALS

G = 9.81
TIMESTEP = 0.0002   # 5 kHz physics


@dataclass
class SceneSpec:
    kind: str            # "drop" (sphere) or "slide" (box)
    surface: str         # material name of the surface
    obj: str             # material name of the object (acoustic only)
    duration: float = 2.5
    drop_height: float = 0.5
    slide_speed: float = 3.0
    seed: int = 0

    @property
    def name(self):
        return f"{self.kind}_{self.obj}_on_{self.surface}"


OBJ_MASS = {"drop": 0.2, "slide": 0.5}
OBJ_SIZE = {"drop": 0.03, "slide": 0.05}     # radius / half-extent (m)


def build_xml(spec: SceneSpec) -> str:
    s = MATERIALS[spec.surface]
    tc, dr = s.solref
    if spec.kind == "slide":
        dr = max(dr, 1.0)    # a sliding block should not hop; avoid solver-induced chatter
    contact = f'friction="{s.friction} 0.005 0.0001" solref="{tc} {dr}"'
    if spec.kind == "drop":
        r = OBJ_SIZE["drop"]
        body = (f'<body name="obj" pos="0 0 {spec.drop_height + r}"><freejoint/>'
                f'<geom name="obj" type="sphere" size="{r}" mass="{OBJ_MASS["drop"]}" {contact}/></body>')
    else:
        h = OBJ_SIZE["slide"]
        hz = 0.4 * h     # flat block: wide/tall ratio of 2.5 keeps it from tipping or rocking at mu ~ 1
        body = (f'<body name="obj" pos="-0.8 0 {hz + 0.001}"><freejoint/>'
                f'<geom name="obj" type="box" size="{h} {h} {hz}" mass="{OBJ_MASS["slide"]}" {contact}/></body>')
    return f"""
<mujoco>
  <option timestep="{TIMESTEP}" gravity="0 0 -{G}"/>
  <worldbody>
    <geom name="surface" type="box" size="2 1 0.05" pos="0 0 -0.05" {contact}/>
    {body}
  </worldbody>
</mujoco>"""


def simulate(spec: SceneSpec):
    model = mujoco.MjModel.from_xml_string(build_xml(spec))
    data = mujoco.MjData(model)
    obj_g = model.geom("obj").id
    surf_g = model.geom("surface").id
    mass = float(model.body("obj").mass[0])
    if spec.kind == "slide":
        data.qvel[0] = spec.slide_speed

    steps = int(spec.duration / TIMESTEP)
    t = np.zeros(steps)
    fn = np.zeros(steps)
    vt = np.zeros(steps)
    z = np.zeros(steps)
    f6 = np.zeros(6)
    vel = np.zeros(6)
    pair = {obj_g, surf_g}
    for i in range(steps):
        mujoco.mj_step(model, data)
        total = 0.0
        for c in range(data.ncon):
            con = data.contact[c]
            if {con.geom1, con.geom2} == pair:
                mujoco.mj_contactForce(model, data, c, f6)
                total += f6[0]                       # normal force (contact frame)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, obj_g, vel, 0)
        t[i] = data.time
        fn[i] = total
        vt[i] = np.hypot(vel[3], vel[4])             # tangential speed (surface is horizontal)
        z[i] = data.xpos[model.body("obj").id][2]
    # MuJoCo's soft contacts make a resting/sliding body micro-hop (contact only activates on
    # penetration), so raw fn flickers between 0 and spikes whose average equals the weight
    # support. A 20 ms moving average recovers that sustained load for friction rendering.
    fn_s = uniform_filter1d(fn, size=int(0.02 / TIMESTEP), mode="nearest")
    return {"t": t, "fn": fn, "fn_s": fn_s, "vt": vt, "z": z, "mass": mass}


def extract_events(tele, f_min=0.05, dv_min=0.3, window=0.02):
    """Turn the force trace into discrete impacts.

    A contact segment is a run of steps with normal force above `f_min`. Its
    impulse is the force in excess of the object's weight, integrated over at
    most `window` seconds from onset, so resting/sliding contact is not
    mistaken for one huge impact. Impacts with J/mass below `dv_min` (m/s) are
    discarded: they are solver micro-hops and inaudible taps.
    """
    t, fn, mass = tele["t"], tele["fn"], tele["mass"]
    dt = t[1] - t[0]
    on = fn > f_min
    edges = np.diff(on.astype(int), prepend=0)
    starts = np.where(edges == 1)[0]
    events = []
    for s in starts:
        e = s
        while e < len(on) and on[e]:
            e += 1
        w_end = min(e, s + int(window / dt))
        J = float(np.sum(np.maximum(fn[s:w_end] - mass * G, 0.0)) * dt)
        if J / mass >= dv_min:
            events.append({"t": float(t[s]), "J": J,
                           "peak_force": float(fn[s:w_end].max()),
                           "contact_ms": float((e - s) * dt * 1e3)})
    return events


# =============================================================================
# Multi-Object Physics (for complex scenarios: stacking, tumbling, collisions)
# =============================================================================

@dataclass
class ObjectSpec:
    """Individual object in a multi-object scene."""
    name: str                                          # Unique ID: "box1", "sphere2", etc.
    shape: str                                         # "sphere", "box", "capsule", "cylinder"
    material: str                                      # Material for physics & acoustic synthesis
    mass: float                                        # kg
    size: Union[float, Tuple[float, float, float]]    # radius (sphere) or (x, y, z) half-extents
    initial_pos: Tuple[float, float, float]           # (x, y, z) initial position
    initial_vel: Optional[Tuple[float, float, float]] = None  # (vx, vy, vz) initial velocity


@dataclass
class MultiSceneSpec:
    """Multi-object physics scenario."""
    kind: str                    # "stack", "tumble", "cascade", "scatter"
    objects: List[ObjectSpec]    # List of objects in the scene
    surface: str                 # Surface material
    duration: float = 2.5        # Simulation duration (s)
    seed: int = 0                # Random seed for modal synthesis

    @property
    def name(self):
        return f"{self.kind}_{self.surface}_{len(self.objects)}obj"


def build_multi_xml(spec: MultiSceneSpec) -> Tuple[str, Dict[str, Dict]]:
    """Build MuJoCo XML for multi-object scene.

    Args:
        spec: Multi-object scene specification

    Returns:
        xml_string: MuJoCo XML scene definition
        obj_registry: {name: {"mass": float, "material": str, "shape": str, "size": ...}}
    """
    surf_mat = MATERIALS[spec.surface]
    tc, dr = surf_mat.solref
    surf_contact = f'friction="{surf_mat.friction} 0.005 0.0001" solref="{tc} {dr}"'

    xml_bodies = []
    obj_registry = {}

    for obj in spec.objects:
        mat = MATERIALS[obj.material]
        obj_tc, obj_dr = mat.solref
        contact = f'friction="{mat.friction} 0.005 0.0001" solref="{obj_tc} {obj_dr}"'

        # Format position
        px, py, pz = obj.initial_pos
        pos_str = f"{px} {py} {pz}"

        # Generate body XML based on shape
        if obj.shape == "sphere":
            r = obj.size if isinstance(obj.size, (int, float)) else obj.size[0]
            body_xml = f'''    <body name="{obj.name}" pos="{pos_str}">
      <freejoint/>
      <geom name="{obj.name}" type="sphere" size="{r}" mass="{obj.mass}" {contact}/>
    </body>'''
            size_info = r

        elif obj.shape == "box":
            if isinstance(obj.size, (int, float)):
                sx, sy, sz = obj.size, obj.size, obj.size
            else:
                sx, sy, sz = obj.size
            body_xml = f'''    <body name="{obj.name}" pos="{pos_str}">
      <freejoint/>
      <geom name="{obj.name}" type="box" size="{sx} {sy} {sz}" mass="{obj.mass}" {contact}/>
    </body>'''
            size_info = (sx, sy, sz)

        elif obj.shape == "capsule":
            # Capsule: size = (radius, half-length)
            if isinstance(obj.size, (int, float)):
                r, hl = obj.size, obj.size * 2
            else:
                r, hl = obj.size if len(obj.size) == 2 else (obj.size[0], obj.size[0] * 2)
            body_xml = f'''    <body name="{obj.name}" pos="{pos_str}">
      <freejoint/>
      <geom name="{obj.name}" type="capsule" size="{r} {hl}" mass="{obj.mass}" {contact}/>
    </body>'''
            size_info = (r, hl)

        elif obj.shape == "cylinder":
            # Cylinder: size = (radius, half-length)
            if isinstance(obj.size, (int, float)):
                r, hl = obj.size, obj.size * 2
            else:
                r, hl = obj.size if len(obj.size) == 2 else (obj.size[0], obj.size[0] * 2)
            body_xml = f'''    <body name="{obj.name}" pos="{pos_str}">
      <freejoint/>
      <geom name="{obj.name}" type="cylinder" size="{r} {hl}" mass="{obj.mass}" {contact}/>
    </body>'''
            size_info = (r, hl)

        else:
            raise ValueError(f"Unsupported shape: {obj.shape}")

        xml_bodies.append(body_xml)
        obj_registry[obj.name] = {
            "mass": obj.mass,
            "material": obj.material,
            "shape": obj.shape,
            "size": size_info,
            "initial_vel": obj.initial_vel
        }

    # Build complete XML
    bodies_str = '\n'.join(xml_bodies)
    xml = f'''<mujoco>
  <option timestep="{TIMESTEP}" gravity="0 0 -{G}"/>
  <worldbody>
    <geom name="surface" type="box" size="2 1 0.05" pos="0 0 -0.05" {surf_contact}/>
{bodies_str}
  </worldbody>
</mujoco>'''

    return xml, obj_registry


def _geom_id_to_name(geom_id: int, obj_registry: Dict, geom_id_map: Dict) -> str:
    """Map geom ID to object name."""
    if geom_id == geom_id_map.get("surface"):
        return "surface"

    for name, info in obj_registry.items():
        if geom_id == info.get("geom_id"):
            return name

    return f"unknown_geom_{geom_id}"


def simulate_multi(spec: MultiSceneSpec) -> Tuple[Dict[Tuple[str, str], Dict], Dict]:
    """Simulate multi-object scene with per-pair contact tracking.

    Args:
        spec: Multi-object scene specification

    Returns:
        telemetry_by_pair: {(name1, name2): {"t": arr, "fn": arr, "ft": arr, "pos": arr}}
        obj_registry: {name: {"geom_id": int, "mass": float, "material": str, ...}}
    """
    xml, obj_registry = build_multi_xml(spec)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Resolve geom IDs after model load
    geom_id_map = {"surface": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "surface")}

    for name in obj_registry:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        obj_registry[name]["geom_id"] = geom_id

    # Set initial velocities
    for i, obj_name in enumerate(obj_registry.keys()):
        if obj_registry[obj_name]["initial_vel"] is not None:
            vx, vy, vz = obj_registry[obj_name]["initial_vel"]
            # Each freejoint has 6 DOF velocities: [vx, vy, vz, wx, wy, wz]
            base_idx = i * 6
            data.qvel[base_idx:base_idx+3] = [vx, vy, vz]

    # Initialize telemetry storage per contact pair
    telemetry = {}  # {(name1, name2): {t: [], fn: [], ft: [], pos: []}}

    steps = int(spec.duration / TIMESTEP)
    f6 = np.zeros(6)

    for i in range(steps):
        mujoco.mj_step(model, data)

        # Process all contacts this timestep
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = con.geom1, con.geom2

            # Look up object names from geom IDs
            name1 = _geom_id_to_name(g1, obj_registry, geom_id_map)
            name2 = _geom_id_to_name(g2, obj_registry, geom_id_map)

            # Normalize pair (consistent ordering: alphabetical)
            pair = tuple(sorted([name1, name2]))

            # Initialize telemetry for new pairs
            if pair not in telemetry:
                telemetry[pair] = {"t": [], "fn": [], "ft": [], "pos": []}

            # Extract contact force
            mujoco.mj_contactForce(model, data, c, f6)
            fn = f6[0]  # Normal force
            ft = np.hypot(f6[1], f6[2])  # Tangential force magnitude

            # Store contact data
            telemetry[pair]["t"].append(data.time)
            telemetry[pair]["fn"].append(fn)
            telemetry[pair]["ft"].append(ft)
            telemetry[pair]["pos"].append(con.pos.copy())

    # Convert lists to arrays and add smoothed forces
    for pair in telemetry:
        for key in ["t", "fn", "ft"]:
            telemetry[pair][key] = np.array(telemetry[pair][key])
        telemetry[pair]["pos"] = np.array(telemetry[pair]["pos"])

        # Add smoothed normal force (for friction rendering compatibility)
        if len(telemetry[pair]["fn"]) > 0:
            telemetry[pair]["fn_s"] = uniform_filter1d(
                telemetry[pair]["fn"],
                size=min(int(0.02 / TIMESTEP), len(telemetry[pair]["fn"])),
                mode="nearest"
            )
        else:
            telemetry[pair]["fn_s"] = np.array([])

    return telemetry, obj_registry


def extract_events_multi(
    telemetry_by_pair: Dict[Tuple[str, str], Dict],
    obj_registry: Dict,
    f_min: float = 0.05,
    dv_min: float = 0.3,
    window: float = 0.02
) -> Dict[Tuple[str, str], List[Dict]]:
    """Extract impact events per contact pair in multi-object scene.

    Args:
        telemetry_by_pair: {(obj1, obj2): tele_dict} from simulate_multi
        obj_registry: Object registry with masses
        f_min: Minimum force threshold (N)
        dv_min: Minimum velocity change (m/s)
        window: Integration window (s)

    Returns:
        {(obj1, obj2): [events with obj labels and contact positions]}
    """
    events_by_pair = {}

    for (name1, name2), tele in telemetry_by_pair.items():
        # Skip if no contact data or insufficient data for event extraction
        if len(tele["t"]) < 2:
            events_by_pair[(name1, name2)] = []
            continue

        # Check if we have valid time steps
        dt = tele["t"][1] - tele["t"][0]
        if dt <= 0 or not np.isfinite(dt):
            events_by_pair[(name1, name2)] = []
            continue

        # Determine mass for impulse→velocity conversion
        # For obj-obj: use reduced mass; for obj-surface: use obj mass
        if name2 == "surface":
            mass = obj_registry[name1]["mass"]
        elif name1 == "surface":
            mass = obj_registry[name2]["mass"]
        else:
            # Reduced mass for obj-obj collision
            m1 = obj_registry[name1]["mass"]
            m2 = obj_registry[name2]["mass"]
            mass = (m1 * m2) / (m1 + m2)

        # Construct telemetry dict in old format for compatibility
        tele_compat = {
            "t": tele["t"],
            "fn": tele["fn"],
            "mass": mass
        }

        # Use existing event extraction
        events = extract_events(tele_compat, f_min, dv_min, window)

        # Label events with object names and contact position
        for j, ev in enumerate(events):
            ev["obj1"] = name1
            ev["obj2"] = name2
            # Find contact position closest to event time
            if len(tele["pos"]) > 0:
                idx = np.argmin(np.abs(tele["t"] - ev["t"]))
                ev["contact_pos"] = tuple(tele["pos"][idx].tolist())
            else:
                ev["contact_pos"] = (0.0, 0.0, 0.0)

        events_by_pair[(name1, name2)] = events

    return events_by_pair


# Spec Reconstruction for Video Rendering
# ==========================================

def reconstruct_scene_spec(episode_name: str, duration: float = 2.5) -> SceneSpec:
    """Reconstruct SceneSpec from episode name.

    Parses episode names like "drop_steel_on_glass" or "slide_wood_on_steel"
    to reconstruct the original scene specification for video rendering.

    Args:
        episode_name: Episode name (e.g., "drop_steel_on_glass")
        duration: Simulation duration in seconds

    Returns:
        SceneSpec object

    Raises:
        ValueError: If episode name format is not recognized

    Example:
        spec = reconstruct_scene_spec("drop_steel_on_glass", duration=2.5)
        xml = build_xml(spec)
        model = mujoco.MjModel.from_xml_string(xml)
    """
    # Parse episode name: "{kind}_{obj}_on_{surface}"
    parts = episode_name.split("_on_")
    if len(parts) != 2:
        raise ValueError(
            f"Cannot parse episode name '{episode_name}'. "
            f"Expected format: 'kind_obj_on_surface' (e.g., 'drop_steel_on_glass')"
        )

    surface = parts[1]

    # Parse kind and obj from first part
    left_parts = parts[0].split("_")
    if len(left_parts) < 2:
        raise ValueError(
            f"Cannot parse episode name '{episode_name}'. "
            f"Expected at least 'kind_obj_on_surface'"
        )

    kind = left_parts[0]  # "drop" or "slide"
    obj = "_".join(left_parts[1:])  # Handle multi-word materials

    # Validate kind
    if kind not in ("drop", "slide"):
        raise ValueError(f"Unknown kind '{kind}'. Expected 'drop' or 'slide'")

    # Validate materials exist
    if obj not in MATERIALS:
        raise ValueError(f"Unknown object material '{obj}'")
    if surface not in MATERIALS:
        raise ValueError(f"Unknown surface material '{surface}'")

    return SceneSpec(
        kind=kind,
        obj=obj,
        surface=surface,
        duration=duration
    )


def reconstruct_multi_scene_spec(
    episode_name: str,
    registry: Dict[str, Dict],
    duration: float = 2.5
) -> MultiSceneSpec:
    """Reconstruct MultiSceneSpec from episode name and object registry.

    Args:
        episode_name: Episode name (e.g., "cascade_steel_3obj")
        registry: Object registry from saved telemetry
                 {name: {"mass": float, "material": str, "shape": str, "size": ...}}
        duration: Simulation duration in seconds

    Returns:
        MultiSceneSpec object

    Raises:
        ValueError: If episode name or registry is invalid

    Example:
        episode = load_episode(episode_dir)
        spec = reconstruct_multi_scene_spec(
            episode.name,
            episode.registry,
            episode.duration
        )
        xml, _ = build_multi_xml(spec)
        model = mujoco.MjModel.from_xml_string(xml)
    """
    if not registry:
        raise ValueError("Registry is required for multi-object reconstruction")

    # Parse episode name: "{kind}_{surface}_{N}obj"
    parts = episode_name.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Cannot parse episode name '{episode_name}'. "
            f"Expected format: 'kind_surface_Nobj' (e.g., 'cascade_steel_3obj')"
        )

    # Last part should be "{N}obj"
    if not parts[-1].endswith("obj"):
        raise ValueError(
            f"Episode name '{episode_name}' should end with 'Nobj' "
            f"(e.g., '3obj')"
        )

    # Extract kind and surface
    kind = parts[0]
    surface = "_".join(parts[1:-1])  # Everything between kind and "Nobj"

    # Validate surface material
    if surface not in MATERIALS:
        raise ValueError(f"Unknown surface material '{surface}'")

    # Reconstruct ObjectSpec list from registry
    objects = []
    for obj_name, obj_info in registry.items():
        # Get required fields
        mass = obj_info.get("mass")
        material = obj_info.get("material")
        shape = obj_info.get("shape")
        size = obj_info.get("size")

        if not all([mass, material, shape, size is not None]):
            raise ValueError(
                f"Incomplete registry for object '{obj_name}'. "
                f"Required: mass, material, shape, size"
            )

        # Reconstruct initial position from telemetry (use default position)
        # For video rendering, exact initial position doesn't matter since
        # we'll be setting positions from telemetry anyway
        initial_pos = (0.0, 0.0, 0.5)  # Default position

        initial_vel = obj_info.get("initial_vel")

        objects.append(ObjectSpec(
            name=obj_name,
            shape=shape,
            material=material,
            mass=mass,
            size=size,
            initial_pos=initial_pos,
            initial_vel=initial_vel
        ))

    return MultiSceneSpec(
        kind=kind,
        objects=objects,
        surface=surface,
        duration=duration
    )
