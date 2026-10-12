# Phase 2A: Spatial Audio - Function Signatures

Scope: `physonic/spatial/**/*.py`

## `physonic/spatial/hrtf.py`

```python
@dataclass
class HRTFSet:
    sample_rate: int
    elevations: np.ndarray
    azimuths: Dict[int, np.ndarray]
    hrir_left: Dict[Tuple[int, int], np.ndarray]
    hrir_right: Dict[Tuple[int, int], np.ndarray]
    hrir_length: int

def load_kemar_hrtf(path: str) -> HRTFSet:
    """Load MIT KEMAR HRTF. Resample to 44.1kHz, truncate to 128 samples."""

def interpolate_hrtf(hrtf: HRTFSet, azim: float, elev: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (hrir_l, hrir_r) using nearest-neighbor interpolation."""

def convolve_hrtf(mono: np.ndarray, hrir_l: np.ndarray, hrir_r: np.ndarray) -> np.ndarray:
    """Return stereo array shape (N+M-1, 2) using scipy.signal.fftconvolve."""
```

## `physonic/spatial/microphone.py`

```python
@dataclass
class BinauralMicrophone:
    position: np.ndarray  # [x,y,z] meters
    orientation: float    # yaw degrees

    def compute_source_angle(self, source_pos: np.ndarray) -> Tuple[float, float]:
        """Return (azimuth, elevation) in degrees."""
```

## `physonic/spatial/renderer.py`

```python
def render_spatial_source(
    mono: np.ndarray,
    source_pos: np.ndarray,
    mic: BinauralMicrophone,
    hrtf: HRTFSet
) -> np.ndarray:
    """Apply 1/r attenuation + HRTF convolution. Return stereo (N, 2)."""
```

## Tests

- ITD within ±0.1 ms for [0°, ±45°, ±90°]
- ILD within ±2 dB @ 4kHz
- Render 1 sec audio in <100ms

See `.claude/commands/validate-spatial.md` for test scripts.
