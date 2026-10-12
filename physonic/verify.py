"""Sanity checks + spectrogram sheet for the generated clips.

    python3 -m physonic.verify --out outputs/mono
    python3 -m physonic.verify --out outputs/spatial --binaural
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

SURFACES = ["steel", "glass", "wood", "rubber"]


def centroid(x, sr):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float((f * spec).sum() / max(spec.sum(), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--binaural", action="store_true", help="Verify binaural (stereo) outputs")
    args = ap.parse_args()
    out = Path(args.out)

    suffix = "_binaural" if args.binaural else ""
    mode = "stereo" if args.binaural else "mono"
    print(f"Verifying {mode} outputs from {out}")
    print(f"{'clip':28s} {'peak':>6s} {'rms dB':>7s} {'centroid':>9s} {'onset err':>10s}")
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), sharex=False)
    for r, kind in enumerate(["drop", "slide"]):
        for c, surf in enumerate(SURFACES):
            obj = "steel" if kind == "drop" else "wood"
            name = f"{kind}_{obj}_on_{surf}"

            # Read audio (mono or stereo)
            x, sr = sf.read(out / f"{name}{suffix}.wav")

            # Convert stereo to mono for analysis (average L/R channels)
            if x.ndim == 2:
                x = x.mean(axis=1)

            ev = json.loads((out / f"{name}_events.json").read_text())
            assert np.isfinite(x).all() and np.abs(x).max() > 0.05, f"{name}: silent or invalid"
            assert np.abs(x).max() < 1.0, f"{name}: clipped"

            rms_db = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
            if ev:   # audio onset (first sample above 5% of the local peak) vs. physics contact time
                t0 = ev[0]["t"]
                lo = int((t0 - 0.05) * sr)
                win = np.abs(x[lo:lo + int(0.2 * sr)])
                hit = np.argmax(win > 0.05 * win.max()) + lo
                onset_err = (hit / sr - t0) * 1e3
                seg = x[int(t0 * sr):int(t0 * sr) + int(0.1 * sr)]
            else:
                onset_err = float("nan")
                seg = x[int(0.1 * sr):int(0.3 * sr)]
            print(f"{name:28s} {np.abs(x).max():6.3f} {rms_db:7.1f} {centroid(seg, sr):9.0f} {onset_err:9.1f}ms")

            f, t, S = spectrogram(x, sr, nperseg=1024, noverlap=768)
            ax = axes[r, c]
            ax.pcolormesh(t, f / 1000, 10 * np.log10(S + 1e-12), vmin=-120, vmax=-40, shading="auto", cmap="magma")
            ax.set_ylim(0, 16)
            ax.set_title(name.replace("_", " "), fontsize=9)
            if c == 0:
                ax.set_ylabel("kHz")
            if r == 1:
                ax.set_xlabel("s")
    title = f"Synthesized audio spectrograms [{mode}] (top: steel ball drop, bottom: wood block slide)"
    fig.suptitle(title)
    fig.tight_layout()
    output_file = out / f"spectrograms{suffix}.png"
    fig.savefig(output_file, dpi=110)
    print(f"wrote {output_file}")


if __name__ == "__main__":
    main()
