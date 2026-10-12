"""Generate synthetic HRTF dataset for testing spatial audio.

This creates a minimal HRTF dataset with realistic ITD/ILD cues for testing
the spatial audio pipeline without needing the full MIT KEMAR dataset.

Usage:
    python3 scripts/setup_hrtf_test.py [--output assets/hrtf/test_kemar]
"""
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path


def generate_synthetic_hrtf(output_dir: Path, sr: int = 44100, hrir_length: int = 128):
    """Generate synthetic HRTF dataset with realistic ITD/ILD cues.

    Args:
        output_dir: Output directory (will create compact/ subdirectory)
        sr: Sample rate (Hz)
        hrir_length: HRIR length in samples
    """
    hrtf_dir = output_dir / "compact"

    # Create dataset with 3 elevations, multiple azimuths per elevation
    elevations = {
        -40: [0, 45, 90, 135, 180, 225, 270, 315],  # Below
        0: [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330],  # Horizontal
        40: [0, 45, 90, 135, 180, 225, 270, 315],   # Above
    }

    print(f"Generating synthetic HRTF dataset in {hrtf_dir}")
    print(f"Sample rate: {sr} Hz, HRIR length: {hrir_length} samples")

    for elevation, azimuths in elevations.items():
        elev_dir = hrtf_dir / f"elev{elevation}"
        elev_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Elevation {elevation:3d}°: {len(azimuths)} azimuths")

        for azim in azimuths:
            # Generate synthetic HRIRs with realistic ITD/ILD cues
            t = np.arange(hrir_length) / sr

            # ITD: Interaural Time Difference
            # Max ITD ~0.7ms at 90° azimuth (sound from side)
            # ITD = head_radius / speed_of_sound * sin(azimuth)
            # Simplified model: linear with azimuth up to 90°
            azim_rad = np.radians(azim)
            itd_sec = 0.0007 * np.sin(azim_rad)  # Max 0.7ms at ±90°
            delay_samp_l = int(abs(itd_sec) * sr)

            # Create impulse responses with realistic spectral content
            hrir_l = np.zeros(hrir_length, dtype=np.float32)
            hrir_r = np.zeros(hrir_length, dtype=np.float32)

            # Base onset time
            onset_base = 10

            # Right ear timing
            if azim_rad > 0:  # Source on left: right ear hears first
                onset_r = onset_base
                onset_l = onset_base + delay_samp_l
            else:  # Source on right or front: similar timing
                onset_r = onset_base
                onset_l = onset_base + delay_samp_l

            # Create shaped impulses with exponential decay
            impulse_dur = 30
            if onset_r + impulse_dur < hrir_length:
                window = np.hanning(impulse_dur)
                decay = np.exp(-t[onset_r:onset_r+impulse_dur] * 200)
                hrir_r[onset_r:onset_r+impulse_dur] = window * decay * 0.15

            if onset_l + impulse_dur < hrir_length:
                window = np.hanning(impulse_dur)
                decay = np.exp(-t[onset_l:onset_l+impulse_dur] * 200)
                hrir_l[onset_l:onset_l+impulse_dur] = window * decay * 0.15

            # ILD: Interaural Level Difference (head shadow effect)
            # Frequency-dependent in reality, simplified here
            # More attenuation for sounds from the side
            shadow_azim = abs(np.sin(azim_rad))  # 0 at front/back, 1 at sides

            if azim_rad > 0:  # Source on left
                hrir_r *= 1.0 - 0.3 * shadow_azim  # Attenuate right ear
            else:  # Source on right
                hrir_l *= 1.0 - 0.3 * shadow_azim  # Attenuate left ear

            # Elevation affects spectral content (pinnae filtering)
            # Simplified: slight high-frequency boost for elevated sources
            if elevation != 0:
                elev_factor = abs(elevation) / 40.0
                # Add slight high-frequency component
                hf_component = np.random.randn(hrir_length) * 0.01 * elev_factor
                hrir_l += hf_component.astype(np.float32)
                hrir_r += hf_component.astype(np.float32)

            # Normalize to prevent clipping
            max_val = max(abs(hrir_l).max(), abs(hrir_r).max())
            if max_val > 0:
                hrir_l /= max_val * 1.2
                hrir_r /= max_val * 1.2

            # Write WAV files (KEMAR naming convention)
            sf.write(elev_dir / f"L{elevation}e{azim:03d}a.wav", hrir_l, sr)
            sf.write(elev_dir / f"R{elevation}e{azim:03d}a.wav", hrir_r, sr)

    print(f"\n✓ Synthetic HRTF dataset generated successfully")
    print(f"  Location: {output_dir}")
    print(f"  Elevations: {list(elevations.keys())}")
    print(f"  Total measurements: {sum(len(azims) for azims in elevations.values())}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic HRTF dataset for testing")
    parser.add_argument("--output", default="assets/hrtf/test_kemar",
                        help="Output directory (default: assets/hrtf/test_kemar)")
    parser.add_argument("--sr", type=int, default=44100,
                        help="Sample rate in Hz (default: 44100)")
    parser.add_argument("--hrir-length", type=int, default=128,
                        help="HRIR length in samples (default: 128)")

    args = parser.parse_args()

    output_dir = Path(args.output)
    generate_synthetic_hrtf(output_dir, args.sr, args.hrir_length)

    print(f"\nTo use this dataset:")
    print(f"  python3 -m physonic.run_demo --spatial --hrtf-path {args.output}")


if __name__ == "__main__":
    main()
