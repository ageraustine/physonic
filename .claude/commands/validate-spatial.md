# Command: /validate-spatial

Run Phase 2A spatial audio validation suite.

## Validation Steps

### 1. Unit Tests
```bash
source .venv/bin/activate
pytest tests/test_spatial/ -v --cov=physonic.spatial --cov-report=term-missing
```

**Pass criteria**: All tests pass, coverage ≥90%

---

### 2. ITD Validation
```bash
python3 << 'EOF'
import numpy as np
from physonic.spatial.hrtf import load_kemar_hrtf, interpolate_hrtf, convolve_hrtf
from physonic.spatial.microphone import BinauralMicrophone
from physonic.spatial.renderer import render_spatial_source

# Load HRTF
hrtf = load_kemar_hrtf("assets/hrtf/mit_kemar/")

# Generate impulse
impulse = np.zeros(1000)
impulse[100] = 1.0

mic = BinauralMicrophone(np.array([0, 0, 1]), 0)

test_cases = [
    (0, 0, 0.0),      # Front
    (90, 0, 0.65),    # Left
    (-90, 0, -0.65),  # Right
    (45, 0, 0.35),
    (-45, 0, -0.35),
]

print("ITD Validation:")
print(f"{'Azimuth':>8s} {'Expected ITD':>12s} {'Measured ITD':>12s} {'Error':>8s} {'Status':>8s}")
print("-" * 60)

all_pass = True
for azim, elev, expected_itd in test_cases:
    # Compute source position
    angle_rad = np.radians(azim)
    dist = 1.0
    source_pos = mic.position + np.array([
        dist * np.cos(angle_rad),
        dist * np.sin(angle_rad),
        0
    ])

    # Render
    stereo = render_spatial_source(impulse, source_pos, mic, hrtf)

    # Measure ITD
    peak_l = np.argmax(np.abs(stereo[:, 0]))
    peak_r = np.argmax(np.abs(stereo[:, 1]))
    itd_ms = (peak_l - peak_r) / 44.1

    error = abs(itd_ms - expected_itd)
    status = "PASS" if error < 0.1 else "FAIL"
    all_pass = all_pass and (status == "PASS")

    print(f"{azim:8.0f}° {expected_itd:11.2f} ms {itd_ms:11.2f} ms {error:7.2f} ms {status:>8s}")

print("\n" + ("="*60))
print(f"ITD Validation: {'PASS' if all_pass else 'FAIL'}")
EOF
```

**Pass criteria**: All 5 test cases within ±0.1 ms

---

### 3. ILD Validation
```bash
python3 << 'EOF'
import numpy as np
from physonic.spatial.hrtf import load_kemar_hrtf
from physonic.spatial.microphone import BinauralMicrophone
from physonic.spatial.renderer import render_spatial_source

hrtf = load_kemar_hrtf("assets/hrtf/mit_kemar/")
mic = BinauralMicrophone(np.array([0, 0, 1]), 0)

# Generate 4 kHz tone (ILD most prominent at high frequencies)
sr = 44100
t = np.arange(0, 0.1, 1/sr)
tone = np.sin(2 * np.pi * 4000 * t)

# Test at 90° (left)
source_pos = mic.position + np.array([0, 1, 0])
stereo = render_spatial_source(tone, source_pos, mic, hrtf)

rms_l = np.sqrt(np.mean(stereo[:, 0] ** 2))
rms_r = np.sqrt(np.mean(stereo[:, 1] ** 2))
ild_db = 20 * np.log10(rms_l / rms_r)

print("ILD Validation (90° azimuth, 4 kHz):")
print(f"  Left RMS:  {rms_l:.6f}")
print(f"  Right RMS: {rms_r:.6f}")
print(f"  ILD:       {ild_db:.1f} dB")
print(f"  Expected:  8-18 dB")

status = "PASS" if 8 <= ild_db <= 18 else "FAIL"
print(f"\nILD Validation: {status}")
EOF
```

**Pass criteria**: ILD between 8-18 dB

---

### 4. Generate Listening Test Stimuli
```bash
python3 -m physonic.run_demo --spatial --out outputs/listening_test/
```

**Generates**:
```
outputs/listening_test/
  drop_steel_on_steel_binaural.wav
  drop_steel_on_glass_binaural.wav
  ... (8 binaural clips)
```

**Manual listening test**:
1. Wear headphones (over-ear, flat response)
2. Play each clip
3. Identify: Left, Center, or Right
4. Record answers

**Pass criteria**: ≥80% correct (at least 6/8 clips)

---

### 5. Performance Benchmark
```bash
python3 << 'EOF'
import time
import numpy as np
from physonic.spatial.hrtf import load_kemar_hrtf
from physonic.spatial.microphone import BinauralMicrophone
from physonic.spatial.renderer import render_spatial_source

hrtf = load_kemar_hrtf("assets/hrtf/mit_kemar/")
mic = BinauralMicrophone(np.array([0, 0, 1]), 0)

# 1 second of audio
mono = np.random.randn(44100)
source_pos = np.array([1, 1, 1])

# Warmup
_ = render_spatial_source(mono, source_pos, mic, hrtf)

# Benchmark
n_runs = 10
start = time.time()
for _ in range(n_runs):
    _ = render_spatial_source(mono, source_pos, mic, hrtf)
elapsed = (time.time() - start) / n_runs * 1000

print(f"Performance Benchmark:")
print(f"  Render time: {elapsed:.1f} ms (avg over {n_runs} runs)")
print(f"  Target:      < 100 ms")
print(f"  Status:      {'PASS' if elapsed < 100 else 'FAIL'}")
EOF
```

**Pass criteria**: < 100 ms per render

---

## Summary

Run all 5 validation steps. Phase 2A is complete when:

- [x] Unit tests: All pass, coverage ≥90%
- [x] ITD validation: All 5 cases within ±0.1 ms
- [x] ILD validation: 8-18 dB at 90°
- [x] Listening test: ≥80% correct identification
- [x] Performance: < 100 ms render time

**On success**: Proceed to Phase 2B (room acoustics)

**On failure**: Check error messages, review `.claude/rules/phase_2a_spec.md`
