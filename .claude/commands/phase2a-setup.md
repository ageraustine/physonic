# Command: /phase2a-setup

Set up Phase 2A spatial audio development environment.

## Steps

### 1. Create directory structure
```bash
mkdir -p physonic/spatial
mkdir -p tests/test_spatial
mkdir -p tests/fixtures/hrtf_mini/compact/elev0
mkdir -p assets/hrtf
touch physonic/spatial/__init__.py
touch physonic/spatial/hrtf.py
touch physonic/spatial/microphone.py
touch physonic/spatial/renderer.py
touch tests/test_spatial/conftest.py
touch tests/test_spatial/test_hrtf.py
touch tests/test_spatial/test_microphone.py
touch tests/test_spatial/test_renderer.py
touch tests/test_spatial/test_validation.py
```

### 2. Install dependencies
```bash
source .venv/bin/activate
pip install resampy scipy soundfile pytest pytest-cov mypy
```

### 3. Download HRTF dataset
```bash
# Option 1: MIT KEMAR (recommended)
curl -o assets/hrtf/kemar.zip https://sound.media.mit.edu/resources/KEMAR/full.zip
unzip assets/hrtf/kemar.zip -d assets/hrtf/mit_kemar/

# Option 2: CIPIC (alternative)
# Download from: https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/
```

### 4. Create minimal test fixture
```python
# Run this to generate mini HRTF dataset for tests
python3 << 'EOF'
import numpy as np
import soundfile as sf
from pathlib import Path

# Generate synthetic HRIRs for testing
sr = 44100
hrir_length = 128

fixture_dir = Path("tests/fixtures/hrtf_mini/compact")
elevations = [0]
azimuths = [0, 45, 90, 135, 180]

for elev in elevations:
    elev_dir = fixture_dir / f"elev{elev}"
    elev_dir.mkdir(parents=True, exist_ok=True)

    for azim in azimuths:
        # Synthetic HRIR with ITD/ILD cues
        t = np.arange(hrir_length) / sr

        # Left ear: delay for positive azimuths
        itd_sec = (azim / 90) * 0.0007  # Max ~0.7ms at 90°
        delay_samp_l = int(itd_sec * sr)
        hrir_l = np.zeros(hrir_length)
        if delay_samp_l < hrir_length - 20:
            hrir_l[delay_samp_l:delay_samp_l+20] = np.hanning(20) * 0.1

        # Right ear: no delay
        hrir_r = np.zeros(hrir_length)
        hrir_r[0:20] = np.hanning(20) * 0.1

        # ILD: attenuate shadowed ear
        ild_factor = np.cos(np.radians(azim))
        hrir_r *= (1 + ild_factor) / 2
        hrir_l *= (1 - ild_factor) / 2

        sf.write(elev_dir / f"L{elev}e{azim:03d}a.wav", hrir_l, sr)
        sf.write(elev_dir / f"R{elev}e{azim:03d}a.wav", hrir_r, sr)

print("Created test fixture: tests/fixtures/hrtf_mini/")
EOF
```

### 5. Verify setup
```bash
pytest tests/test_spatial/test_hrtf.py::test_load_kemar_hrtf -v
python3 -c "from physonic.spatial.hrtf import load_kemar_hrtf; print('Import OK')"
```

## Expected Output

```
Created directories:
  physonic/spatial/
  tests/test_spatial/
  tests/fixtures/hrtf_mini/

Installed packages:
  resampy==0.4.2
  scipy==1.11.x
  soundfile==0.12.x
  pytest==7.x
  mypy==1.x

Downloaded HRTF:
  assets/hrtf/mit_kemar/ (200 MB, ~700 measurements)

Test fixture:
  tests/fixtures/hrtf_mini/ (5 azimuths × 1 elevation = 10 WAV files)

Setup complete! Ready to implement Phase 2A.
```

## Next Steps

1. Read `.claude/rules/phase_2a_spec.md`
2. Implement `physonic/spatial/hrtf.py` (start with `load_kemar_hrtf`)
3. Write tests as you go
4. Run `/validate-spatial` when done
