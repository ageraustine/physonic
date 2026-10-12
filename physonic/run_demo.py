"""Run demo: simulate -> extract impacts -> synthesize -> write WAV/JSON/PNG.

    # Mono (contact) audio only
    python3 -m physonic.run_demo [--out outputs] [--seed 0]

    # Binaural (spatial) audio
    python3 -m physonic.run_demo --spatial --hrtf-path assets/hrtf/mit_kemar/ [--mic-position 0,0,1.5]

    # Spatial + room acoustics (preset)
    python3 -m physonic.run_demo --spatial --room warehouse_large

    # Spatial + room acoustics (custom)
    python3 -m physonic.run_demo --spatial --room-dim 30,20,8 --rt60 2.5 --absorption 0.15
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from physonic.contact import modal_synth as ms
from physonic.contact.materials import MATERIALS
from physonic.contact.rig import OBJ_MASS, OBJ_SIZE, SceneSpec, extract_events, simulate

SURFACES = ["steel", "glass", "wood", "rubber"]


def render_scene(spec: SceneSpec, spatial_config: Optional[dict] = None, room_config: Optional[dict] = None):
    """Render contact audio scene with optional spatial rendering and room acoustics.

    Args:
        spec: Scene specification
        spatial_config: Optional dict with 'hrtf', 'mic_position', 'source_position'
        room_config: Optional dict with 'room_spec', 'enable_early', 'enable_late', 'wet_mix'

    Returns:
        (audio, tele, events) where audio is shape (N,) for mono or (N, 2) for stereo
    """
    # 1. Physics simulation
    tele = simulate(spec)
    events = extract_events(tele)

    # 2. Modal synthesis (mono)
    rng = np.random.default_rng(spec.seed)
    surf_mat, obj_mat = MATERIALS[spec.surface], MATERIALS[spec.obj]
    surf_bank = ms.sample_modes(surf_mat, rng, size=1.0)
    obj_bank = ms.sample_modes(obj_mat, rng, size=OBJ_SIZE[spec.kind] / 0.05)

    out = np.zeros(int(spec.duration * ms.SR))
    mass = OBJ_MASS[spec.kind]
    for ev in events:
        ms.render_impact(out, ev["t"], ev["J"], mass, surf_bank, surf_mat, weight=1.0)
        ms.render_impact(out, ev["t"], ev["J"], mass, obj_bank, obj_mat, weight=0.6)
    if spec.kind == "slide":
        out += ms.render_friction(tele["t"], tele["fn_s"], tele["vt"], surf_bank, rng, spec.duration)

    out = ms.limit(out).astype(np.float32)

    # 3. Room acoustics (BEFORE spatial rendering for physical accuracy)
    if room_config is not None:
        from physonic.acoustics import apply_room_acoustics

        room_spec = room_config['room_spec']
        # Get source/listener positions (use spatial config if available, otherwise defaults)
        source_pos = spatial_config['source_position'] if spatial_config else np.array([0.0, 0.0, 0.5])
        mic_pos = spatial_config['mic_position'] if spatial_config else np.array([0.0, 0.0, 1.5])

        out = apply_room_acoustics(
            out,
            source_pos,
            mic_pos,
            room_spec,
            enable_early=room_config.get('enable_early', True),
            enable_late=room_config.get('enable_late', True),
            wet_mix=room_config.get('wet_mix', 0.3)
        )

    # 4. Spatial rendering (on reverb-processed mono)
    if spatial_config is not None:
        from physonic.spatial import BinauralMicrophone, render_spatial_source

        hrtf = spatial_config['hrtf']
        mic = BinauralMicrophone(
            position=spatial_config['mic_position'],
            orientation=0.0
        )
        source_pos = spatial_config['source_position']

        # Render mono → binaural stereo
        out = render_spatial_source(out, source_pos, mic, hrtf)

        # Normalize stereo output to prevent clipping
        peak = np.abs(out).max()
        if peak > 0.99:
            out = out * (0.99 / peak)
        out = out.astype(np.float32)

    return out, tele, events


def main():
    ap = argparse.ArgumentParser(description="Generate contact + spatial audio from physics simulation")
    ap.add_argument("--out", default="outputs", help="Output directory")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")

    # Spatial audio options
    ap.add_argument("--spatial", action="store_true", help="Enable binaural spatial audio")
    ap.add_argument("--hrtf-path", default="assets/hrtf/mit_kemar/",
                    help="Path to HRTF dataset (required if --spatial)")
    ap.add_argument("--mic-position", default="0,0,1.5",
                    help="Microphone position as 'x,y,z' in meters")

    # Room acoustics options
    ap.add_argument("--room", help="Room preset (warehouse_small, warehouse_large, data_center, supermarket, office, etc)")
    ap.add_argument("--room-dim", help="Custom room dimensions as 'L,W,H' in meters (overrides --room)")
    ap.add_argument("--rt60", type=float, help="Custom RT60 reverberation time in seconds (overrides preset)")
    ap.add_argument("--absorption", type=float, help="Wall absorption coefficient 0-1 (overrides preset)")
    ap.add_argument("--no-early", action="store_true", help="Disable early reflections (image-source method)")
    ap.add_argument("--no-late", action="store_true", help="Disable late diffuse reverb (Schroeder)")
    ap.add_argument("--wet-mix", type=float, default=0.3, help="Wet/dry mix for reverb (0-1, default 0.3)")

    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load HRTF if spatial rendering requested
    spatial_config = None
    if args.spatial:
        from physonic.spatial import load_kemar_hrtf

        hrtf_path = Path(args.hrtf_path)
        if not hrtf_path.exists():
            print(f"ERROR: HRTF dataset not found at {hrtf_path}")
            print("Download MIT KEMAR HRTF or run: /phase2a-setup")
            return 1

        print(f"Loading HRTF from {hrtf_path}...")
        hrtf = load_kemar_hrtf(str(hrtf_path))
        mic_pos = np.array([float(x) for x in args.mic_position.split(',')])

        spatial_config = {
            'hrtf': hrtf,
            'mic_position': mic_pos,
        }
        print(f"Spatial audio enabled: mic at {mic_pos}, HRTF loaded ✓")

    # Load room acoustics if requested
    room_config = None
    if args.room or args.room_dim:
        from physonic.acoustics import ROOMS, RoomSpec

        # Start with preset or create custom
        if args.room_dim:
            # Custom dimensions
            dims = tuple(float(x) for x in args.room_dim.split(','))
            if len(dims) != 3:
                print(f"ERROR: --room-dim must be 'L,W,H' (got {args.room_dim})")
                return 1

            # Use defaults if not specified
            rt60 = args.rt60 if args.rt60 is not None else 1.0
            absorption = args.absorption if args.absorption is not None else 0.3

            room_spec = RoomSpec(
                dimensions=dims,
                absorption=absorption,
                rt60=rt60,
                max_reflection_order=2
            )
            print(f"Custom room: {dims[0]}×{dims[1]}×{dims[2]}m, RT60={rt60:.1f}s, α={absorption:.2f}")

        elif args.room:
            # Load preset
            if args.room not in ROOMS:
                print(f"ERROR: Unknown room preset '{args.room}'")
                print(f"Available rooms: {', '.join(ROOMS.keys())}")
                return 1

            room_spec = ROOMS[args.room]

            # Allow overrides
            if args.rt60 is not None or args.absorption is not None:
                from dataclasses import replace
                room_spec = replace(
                    room_spec,
                    rt60=args.rt60 if args.rt60 is not None else room_spec.rt60,
                    absorption=args.absorption if args.absorption is not None else room_spec.absorption
                )

            print(f"Room preset '{args.room}': {room_spec.dimensions[0]}×{room_spec.dimensions[1]}×{room_spec.dimensions[2]}m, RT60={room_spec.rt60:.1f}s")

        room_config = {
            'room_spec': room_spec,
            'enable_early': not args.no_early,
            'enable_late': not args.no_late,
            'wet_mix': args.wet_mix
        }

    specs = [SceneSpec("drop", s, "steel", seed=args.seed) for s in SURFACES]
    specs += [SceneSpec("slide", s, "wood", seed=args.seed) for s in SURFACES]

    summary = {}
    for spec in specs:
        # Compute source position from scene spec
        if spatial_config is not None:
            if spec.kind == "drop":
                source_pos = np.array([0.0, 0.0, spec.drop_height])
            else:  # slide
                source_pos = np.array([-0.8, 0.0, 0.4 * OBJ_SIZE["slide"]])
            spatial_config['source_position'] = source_pos

        audio, tele, events = render_scene(spec, spatial_config, room_config)

        # Save audio (mono or stereo)
        suffix = "_binaural" if args.spatial else ""
        if room_config is not None:
            suffix += "_reverb"
        sf.write(out_dir / f"{spec.name}{suffix}.wav", audio, ms.SR)

        # Save telemetry and events
        np.savez_compressed(out_dir / f"{spec.name}_telemetry.npz",
                            t=tele["t"], fn=tele["fn"], vt=tele["vt"], z=tele["z"])
        (out_dir / f"{spec.name}_events.json").write_text(json.dumps(events, indent=1))

        summary[spec.name] = {"n_events": len(events), "events": events[:6]}

        # Print status
        channels = "stereo" if audio.ndim == 2 else "mono"
        peak = np.abs(audio).max()
        print(f"{spec.name:28s} events={len(events):3d} peak={peak:.3f} [{channels}]")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
