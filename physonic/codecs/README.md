## Neural Audio Codecs

This module provides neural audio codec engines for efficient audio compression in RL environments.

### Available Codecs

#### Mimi (Kyutai)
- **Bitrate**: 1.1 kbps
- **Frame rate**: 12.5 Hz
- **Compression**: ~353x (44.1kHz stereo → 1.1kbps codes)
- **Model**: `kyutai/mimi` on Hugging Face
- **Paper**: [Moshi: A Speech-Text Foundation Model](http://kyutai.org/Moshi.pdf)

### Installation

**Requirements**:
- Python 3.10+
- PyTorch 2.2+
- transformers 4.44-4.45 (Mimi support without PyTorch 2.5 requirement)
- torchaudio 2.2+

**Installation**:
```bash
pip install torch==2.2.2 torchaudio==2.2.2
pip install "transformers>=4.44.0,<4.46.0"
```

**Note**: transformers 4.46+ requires PyTorch 2.5+ which is not yet available for macOS x86_64 (Intel Macs). The 4.44-4.45 range provides Mimi support with PyTorch 2.2.2 compatibility for all platforms.

### Usage

#### Basic Usage (Sequential Mode)

```python
from physonic.codecs import MimiAudioEngine
import numpy as np

# Create codec (lazy-loaded)
codec = MimiAudioEngine()

# Generate test audio (1s, 44.1kHz, stereo)
sample_rate = 44100
duration = 1.0
t = np.linspace(0, duration, int(sample_rate * duration))
audio = np.stack([
    0.5 * np.sin(2 * np.pi * 440 * t),  # Left: A4
    0.5 * np.sin(2 * np.pi * 880 * t),  # Right: A5
])

# Encode (stereo → mono downmix, resample to 24kHz)
encoded = codec.encode(audio, sample_rate)

# Decode back to audio
decoded_audio, decoded_sr = codec.decode(encoded)

print(f"Original: {audio.shape} @ {sample_rate}Hz")
print(f"Codes: {encoded['codes'][0].shape} @ 12.5Hz")
print(f"Decoded: {decoded_audio.shape} @ {decoded_sr}Hz")
print(f"Compression: {audio.size / encoded['codes'][0].size:.1f}x")
```

#### Parallel Mode (Faster)

```python
# Use 25% of CPUs for parallel processing (~20% speedup)
codec = MimiAudioEngine(parallel=True, cpu_percent=25)

# Process longer audio efficiently
long_audio = np.random.randn(2, 44100 * 30)  # 30 seconds
encoded = codec.encode(long_audio, 44100)
```

#### Custom Chunk Sizes

```python
# Override chunk lengths
codec = MimiAudioEngine(
    encode_chunk_length=10.0,  # 10s chunks for encoding
    decode_chunk_length=5.0,   # 5s chunks for decoding
)

# Disable chunking (process entire audio at once)
codec = MimiAudioEngine(
    encode_chunk_length=0,  # 0 or negative = no chunking
    decode_chunk_length=0,
)
```

### Performance

**Benchmark (30s audio, 44.1kHz stereo)**:
- Sequential: ~7.1s (default)
- Parallel (25% CPU): ~5.7s (~20% faster)

**Memory footprint**:
- Original: 2 channels × 44,100 Hz × 30s = 2,646,000 samples
- Mimi codes: 8 codebooks × 12.5 Hz × 30s = 3,000 codes
- Compression: ~882x

### RL Integration

#### Compress Audio Observations

```python
from physonic.envs import SoundLocalizationEnv
from physonic.codecs import MimiAudioEngine

# Create environment
env = SoundLocalizationEnv(use_mel_spectrogram=False)  # Use raw audio

# Create codec
codec = MimiAudioEngine()

# Get observation
obs, info = env.reset()
raw_audio = obs  # Shape: (samples, 2)

# Encode to compressed representation
encoded = codec.encode(raw_audio.T, 44100)
codes = encoded["codes"][0]  # Shape: (8, time_frames)

print(f"Raw audio: {raw_audio.size} samples")
print(f"Compressed: {codes.size} codes")
print(f"Reduction: {raw_audio.size / codes.size:.1f}x")

# Decode when needed (e.g., for rendering)
decoded_audio, sr = codec.decode(encoded)
```

#### Use in Custom RL Environment

```python
class CompressedAudioEnv(gym.Env):
    def __init__(self):
        self.codec = MimiAudioEngine()
        # Observation space: Mimi codes instead of raw audio
        # 8 codebooks × ~6 frames (0.5s @ 12.5Hz)
        self.observation_space = gym.spaces.Box(
            low=0, high=2048,  # Codebook size
            shape=(8, 6),
            dtype=np.int64
        )

    def _get_audio_observation(self):
        # Get raw audio from environment
        raw_audio = self._render_binaural_audio()

        # Encode to compressed codes
        encoded = self.codec.encode(raw_audio, 44100)
        codes = encoded["codes"][0]

        return codes
```

### Technical Details

#### Context-Primed Chunking

Mimi uses causal transformers, so only the left edge of chunks is degraded:
- Each chunk includes CONTEXT_FRAMES (8 frames = 0.64s) of left context
- Context output is discarded, only new frames are kept
- Linear crossfade (10ms) at decode seams for smooth transitions

#### Processing Modes

**Sequential (workers=1)**:
- Encode chunks: 6.0s
- Decode chunks: 2.0s
- Reliable, no system impact

**Parallel (workers=2-4)**:
- Encode chunks: 4.0s
- Decode chunks: 4.0s
- ~20% faster with 25% CPU usage

### Testing

Run the test script:
```bash
python test_mimi_codec.py
```

Expected output:
```
Testing Mimi Codec (Sequential Mode)
  Input: (2, 44100) @ 44100Hz
  Codes: (8, 13) @ 12.5Hz
  Compression: 846.2x
  Reconstruction MSE: 0.000123
  ✓ Test passed!
```
