"""Material table: contact (physics) and acoustic (modal synthesis) parameters.

Acoustic values are plausible engineering approximations for a small
(~10 cm) object, not measured data. Replace with measured or mesh-derived
modal parameters when fidelity matters.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    # --- physics (MuJoCo contact) ---
    friction: float          # sliding friction coefficient
    solref: tuple            # (timeconst [s], dampratio); lower dampratio = bouncier
    # --- acoustics (modal synthesis) ---
    f0: float                # fundamental of a reference object (Hz)
    ratios: tuple            # mode frequency ratios (inharmonic for bars/plates)
    t60: float               # decay time of the fundamental, 60 dB (s)
    damp_slope: float        # extra damping per kHz (high modes die faster)
    gain: float              # radiation efficiency (relative loudness)
    tc_ref: float            # Hertz contact time at 1 m/s, 0.1 kg (s); sets brightness


# Free-free bar mode ratios (Euler-Bernoulli): 1, 2.756, 5.404, 8.933, 13.34, 18.64
_BAR = (1.0, 2.756, 5.404, 8.933, 13.344, 18.64)
# Looser, denser spectrum typical of damped composites / wood
_WOOD = (1.0, 1.9, 3.1, 4.6, 6.4, 8.7)

MATERIALS = {
    "steel":  Material("steel",  0.60, (0.0030, 0.15), 1400.0, _BAR,  1.20, 0.30, 1.00, 6e-5),
    "glass":  Material("glass",  0.40, (0.0030, 0.30), 2200.0, _BAR,  0.90, 0.50, 0.85, 8e-5),
    "wood":   Material("wood",   0.50, (0.0050, 0.50),  600.0, _WOOD, 0.12, 1.50, 0.60, 4e-4),
    "rubber": Material("rubber", 1.10, (0.0120, 0.60),  180.0, _WOOD, 0.04, 2.00, 0.35, 2e-3),
}
