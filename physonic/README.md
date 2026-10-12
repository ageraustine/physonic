# physonic (MVP, Phase 1)

MuJoCo physics -> contact telemetry -> procedural modal-synthesis audio.

```
python3 -m physonic.run_demo --out outputs   # simulate + synthesize 8 clips
python3 -m physonic.verify   --out outputs   # checks + spectrograms.png
```

Requires: `mujoco numpy scipy soundfile matplotlib`.

## Pipeline

1. `rig.py` builds a scene (steel ball dropped 0.5 m, or wood block sliding at 3 m/s)
   on a steel / glass / wood / rubber surface and logs normal force, tangential
   speed and height at 5 kHz.
2. `extract_events` turns the force trace into discrete impacts: onset time and
   impulse J = integral of F dt (excess over the object's weight, first 20 ms).
3. `modal_synth.py` renders each impact as a bank of decaying sinusoids. J sets
   amplitude; material sets frequencies / damping; Hertz contact time low-passes
   the excitation (soft materials sound duller). Sliding adds noise whose
   envelope follows sqrt(normal force) x sliding speed, filtered through the
   same resonators.
4. Outputs per clip: `.wav` (44.1 kHz mono), `_events.json`, `_telemetry.npz`.

## Known limitations (deliberate for the MVP)

- Material acoustic parameters are plausible approximations, not measured data;
  mode frequencies are per-material ratios with random jitter, not derived from mesh geometry.
- Contact physics parameters come from the surface material only (the object
  material affects acoustics, not contact stiffness).
- MuJoCo soft contacts make resting bodies micro-hop. Handled by smoothing the
  normal force for friction and discarding impacts below 0.3 m/s equivalent speed.
- Mono audio, no spatialization / room acoustics yet.
- No LeRobotDataset export yet (Phase 3).
