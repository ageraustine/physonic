"""Run multi-object physics scenarios and synthesize acoustic output.

Usage:
    # List available scenarios
    python3 -m physonic.run_demo_multi --list

    # Run a specific scenario
    python3 -m physonic.run_demo_multi --scenario stack_collapse --surface wood --out outputs/multi

    # Run with spatial audio
    python3 -m physonic.run_demo_multi --scenario two_sphere_collision --spatial --hrtf-path assets/hrtf/test_kemar --out outputs/multi_spatial

    # Custom number of objects (for applicable scenarios)
    python3 -m physonic.run_demo_multi --scenario sphere_cascade --n-objects 8 --out outputs/cascade
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from physonic.contact import modal_synth as ms
from physonic.contact.materials import MATERIALS
from physonic.contact.rig import simulate_multi, extract_events_multi
from physonic.contact.scenarios import SCENARIOS, list_scenarios


def render_multi_scene(spec, spatial_config: Optional[dict] = None):
    """Render multi-object scene with contact audio synthesis.

    Args:
        spec: MultiSceneSpec describing the scene
        spatial_config: Optional dict with 'hrtf', 'mic_position' for binaural rendering

    Returns:
        (audio, telemetry, events_by_pair)
        audio: mono (N,) or stereo (N, 2) depending on spatial_config
        telemetry: {(obj1, obj2): tele_dict}
        events_by_pair: {(obj1, obj2): [events]}
    """
    # 1. Physics simulation
    telemetry, obj_registry = simulate_multi(spec)
    events_by_pair = extract_events_multi(telemetry, obj_registry)

    # 2. Modal synthesis - pre-sample mode banks for each unique material
    rng = np.random.default_rng(spec.seed)
    mode_banks = {}

    for obj_name, obj_info in obj_registry.items():
        mat_name = obj_info["material"]
        if mat_name not in mode_banks:
            mat = MATERIALS[mat_name]
            # Size normalization: use object's size relative to default 0.05m
            size = obj_info["size"]
            if isinstance(size, tuple):
                # For box/capsule/cylinder: use geometric mean of dimensions
                size_factor = (np.prod(size) ** (1/len(size))) / 0.05
            else:
                # For sphere: use radius
                size_factor = size / 0.05

            mode_banks[mat_name] = ms.sample_modes(mat, rng, size=size_factor)

    # 3. Render all impacts
    out = np.zeros(int(spec.duration * ms.SR))

    for (name1, name2), events in events_by_pair.items():
        for ev in events:
            # Both objects vibrate from collision (Newton's third law)

            # Object 1 contribution
            if name1 != "surface":
                mat1 = MATERIALS[obj_registry[name1]["material"]]
                bank1 = mode_banks[obj_registry[name1]["material"]]
                mass1 = obj_registry[name1]["mass"]
                ms.render_impact(out, ev["t"], ev["J"], mass1, bank1, mat1, weight=1.0)

            # Object 2 contribution
            if name2 != "surface":
                mat2 = MATERIALS[obj_registry[name2]["material"]]
                bank2 = mode_banks[obj_registry[name2]["material"]]
                mass2 = obj_registry[name2]["mass"]
                # Slightly lower weight for second object to avoid doubling amplitude
                ms.render_impact(out, ev["t"], ev["J"], mass2, bank2, mat2, weight=0.7)

    out = ms.limit(out).astype(np.float32)

    # 4. Spatial rendering (optional)
    if spatial_config is not None:
        # TODO: Multi-source spatial rendering
        # For now, render from scene center
        from physonic.spatial import BinauralMicrophone, render_spatial_source

        hrtf = spatial_config['hrtf']
        mic = BinauralMicrophone(
            position=spatial_config['mic_position'],
            orientation=0.0
        )
        # Use center of scene as approximate source position
        source_pos = np.array([0.0, 0.0, 0.5])

        out = render_spatial_source(out, source_pos, mic, hrtf)

        # Normalize stereo output
        peak = np.abs(out).max()
        if peak > 0.99:
            out = out * (0.99 / peak)
        out = out.astype(np.float32)

    return out, telemetry, events_by_pair


def main():
    ap = argparse.ArgumentParser(description="Generate multi-object contact audio from physics")
    ap.add_argument("--scenario", help="Scenario name (use --list to see options)")
    ap.add_argument("--list", action="store_true", help="List available scenarios")
    ap.add_argument("--surface", default="wood", help="Surface material (default: wood)")
    ap.add_argument("--n-objects", type=int, help="Number of objects (for applicable scenarios)")
    ap.add_argument("--out", default="outputs_multi", help="Output directory")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")

    # Spatial audio options
    ap.add_argument("--spatial", action="store_true", help="Enable binaural spatial audio")
    ap.add_argument("--hrtf-path", default="assets/hrtf/test_kemar/",
                    help="Path to HRTF dataset (required if --spatial)")
    ap.add_argument("--mic-position", default="0,0,1.5",
                    help="Microphone position as 'x,y,z' in meters")

    args = ap.parse_args()

    # List scenarios
    if args.list:
        print("Available scenarios:")
        for name, desc in list_scenarios().items():
            print(f"  {name:22s} - {desc}")
        return 0

    # Validate scenario
    if not args.scenario:
        print("ERROR: Must specify --scenario (use --list to see options)")
        return 1

    if args.scenario not in SCENARIOS:
        print(f"ERROR: Unknown scenario '{args.scenario}'")
        print(f"Available: {', '.join(SCENARIOS.keys())}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load HRTF if spatial rendering requested
    spatial_config = None
    if args.spatial:
        from physonic.spatial import load_kemar_hrtf

        hrtf_path = Path(args.hrtf_path)
        if not hrtf_path.exists():
            print(f"ERROR: HRTF dataset not found at {hrtf_path}")
            print("Generate test HRTF: python3 scripts/setup_hrtf_test.py")
            return 1

        print(f"Loading HRTF from {hrtf_path}...")
        hrtf = load_kemar_hrtf(str(hrtf_path))
        mic_pos = np.array([float(x) for x in args.mic_position.split(',')])

        spatial_config = {
            'hrtf': hrtf,
            'mic_position': mic_pos,
        }
        print(f"Spatial audio enabled: mic at {mic_pos}, HRTF loaded ✓")

    # Build scenario
    scenario_func = SCENARIOS[args.scenario]

    # Handle n_objects parameter
    spec_kwargs = {"surface": args.surface, "seed": args.seed}
    if args.n_objects is not None:
        # Check if scenario accepts n_objects/n_spheres/n_dominoes
        import inspect
        sig = inspect.signature(scenario_func)
        param_names = list(sig.parameters.keys())

        if "n_objects" in param_names:
            spec_kwargs["n_objects"] = args.n_objects
        elif "n_spheres" in param_names:
            spec_kwargs["n_spheres"] = args.n_objects
        elif "n_dominoes" in param_names:
            spec_kwargs["n_dominoes"] = args.n_objects
        elif "n_cylinders" in param_names:
            spec_kwargs["n_cylinders"] = args.n_objects
        else:
            print(f"WARNING: Scenario '{args.scenario}' does not accept --n-objects parameter")

    spec = scenario_func(**spec_kwargs)

    print(f"\nSimulating: {args.scenario}")
    print(f"  Surface: {args.surface}")
    print(f"  Objects: {len(spec.objects)}")
    print(f"  Duration: {spec.duration}s")

    # Render scene
    audio, telemetry, events_by_pair = render_multi_scene(spec, spatial_config)

    # Save audio
    suffix = "_binaural" if args.spatial else ""
    audio_path = out_dir / f"{spec.name}{suffix}.wav"
    sf.write(audio_path, audio, ms.SR)

    # Save telemetry (per-pair)
    tele_path = out_dir / f"{spec.name}_telemetry.npz"
    tele_arrays = {}
    for (name1, name2), tele in telemetry.items():
        prefix = f"{name1}_{name2}"
        for key, val in tele.items():
            if isinstance(val, np.ndarray):
                tele_arrays[f"{prefix}_{key}"] = val
    np.savez_compressed(tele_path, **tele_arrays)

    # Save events (aggregate all pairs)
    events_all = []
    for (name1, name2), events in events_by_pair.items():
        events_all.extend(events)

    # Sort by time
    events_all.sort(key=lambda e: e["t"])

    events_path = out_dir / f"{spec.name}_events.json"
    events_path.write_text(json.dumps(events_all, indent=1))

    # Print summary
    channels = "stereo" if audio.ndim == 2 else "mono"
    peak = np.abs(audio).max()
    n_events = len(events_all)

    print(f"\n{'Scene':30s} {'Events':>8s} {'Peak':>7s} {'Mode':>7s}")
    print(f"{spec.name:30s} {n_events:8d} {peak:7.3f} {channels:>7s}")

    # Per-pair event breakdown
    print(f"\nEvents per contact pair:")
    for (name1, name2), events in sorted(events_by_pair.items()):
        if len(events) > 0:
            print(f"  {name1:15s} ↔ {name2:15s}: {len(events):3d} impacts")

    print(f"\nWrote:")
    print(f"  {audio_path}")
    print(f"  {tele_path}")
    print(f"  {events_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
