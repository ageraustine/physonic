"""Procedural acoustic synthesis driven by physics telemetry.

Impacts: each contact event (onset time, impulse J = integral of F dt) excites a
bank of decaying sinusoids (modal synthesis). J sets mode amplitudes; material
sets frequencies and damping; contact hardness low-passes the excitation.

Friction: sliding speed and normal force set the envelope of noise that is
filtered through the same resonators (rough-surface excitation model).
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .materials import Material

SR = 44100
AMP = 0.25          # global gain before the soft limiter
FRIC_AMP = 0.012    # friction envelope gain
FRIC_BW = 6.0       # friction resonators are broadened vs. impact ringing


@dataclass
class ModeBank:
    freqs: np.ndarray   # Hz
    decay: np.ndarray   # amplitude decay rate (1/s)
    gains: np.ndarray   # relative excitation


def sample_modes(mat: Material, rng: np.random.Generator, size: float = 1.0) -> ModeBank:
    """Sample a per-object mode bank. `size` ~ characteristic length scale
    relative to the reference object (frequencies scale as 1/size)."""
    f0 = mat.f0 / size * rng.lognormal(0.0, 0.06)
    ratios = np.array(mat.ratios) * rng.normal(1.0, 0.015, len(mat.ratios))
    freqs = f0 * ratios
    freqs = freqs[freqs < 0.45 * SR]
    n = len(freqs)
    t60 = mat.t60 / (1.0 + mat.damp_slope * freqs / 1000.0)
    decay = np.log(1000.0) / t60                    # 60 dB = factor 1000 in amplitude
    gains = rng.uniform(0.6, 1.0, n) / (1.0 + 0.25 * np.arange(n))
    return ModeBank(freqs, decay, gains)


def render_impact(out, t0, J, mass, bank: ModeBank, mat: Material, weight=1.0, sr=SR):
    """Add one impact's modal response into `out` (in place)."""
    i0 = int(round(t0 * sr))
    if i0 >= len(out):
        return
    n = min(int(sr * mat.t60 * 1.1) + 1, len(out) - i0)
    t = np.arange(n) / sr
    dv = max(J / mass, 1e-3)                                        # ~ change in velocity
    tc = mat.tc_ref * dv ** -0.2 * (mass / 0.1) ** (1.0 / 3.0)      # Hertz contact time
    fc = 1.0 / (2.0 * tc)
    rolloff = 1.0 / np.sqrt(1.0 + (bank.freqs / fc) ** 2)           # contact low-pass
    amps = AMP * weight * mat.gain * J * bank.gains * rolloff
    seg = (amps[:, None]
           * np.exp(-bank.decay[:, None] * t[None, :])
           * np.sin(2 * np.pi * bank.freqs[:, None] * t[None, :])).sum(axis=0)
    out[i0:i0 + n] += seg


def _unit(x):
    s = np.std(x)
    return x / s if s > 0 else x


def render_friction(t_phys, fn, vt, bank: ModeBank, rng, duration, sr=SR):
    """Friction/sliding noise from per-step normal force and sliding speed."""
    n = int(duration * sr)
    t_audio = np.arange(n) / sr
    fn_a = np.interp(t_audio, t_phys, fn)
    vt_a = np.interp(t_audio, t_phys, vt)
    env = FRIC_AMP * np.sqrt(np.maximum(fn_a, 0.0)) * np.minimum(vt_a, 4.0)
    env *= np.clip(vt_a / 0.05, 0.0, 1.0)                           # fade out as it stops
    env = lfilter([0.02], [1, -0.98], env)                          # ~ 2 ms smoothing

    noise = rng.standard_normal(n)
    sig = np.zeros(n)
    for f, d, g in zip(bank.freqs, bank.decay, bank.gains):
        r = np.exp(-max(d * FRIC_BW, 80.0) / sr)
        w = 2 * np.pi * f / sr
        sig += g * lfilter([1 - r], [1, -2 * r * np.cos(w), r * r], noise)
    sig = _unit(sig)

    a = np.exp(-2 * np.pi * 2000.0 / sr)                            # broadband scrape
    broad = _unit(lfilter([1 - a], [1, -a], rng.standard_normal(n)))

    b = np.exp(-2 * np.pi * 40.0 / sr)                              # stick-slip roughness
    mod = _unit(lfilter([1 - b], [1, -b], rng.standard_normal(n)))
    mod = np.clip(1.0 + 0.3 * mod, 0.2, None)

    return env * mod * (sig + 0.35 * broad)


def limit(x):
    return np.tanh(x)
