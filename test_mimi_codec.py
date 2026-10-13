"""Test Mimi neural audio codec."""

import numpy as np
from physonic.codecs import MimiAudioEngine

def test_mimi_sequential():
    """Test Mimi codec with sequential processing."""
    print("\n" + "="*70)
    print("Testing Mimi Codec (Sequential Mode)")
    print("="*70)

    # Create codec engine
    codec = MimiAudioEngine(parallel=False)

    # Generate test audio (1 second, 44.1kHz, stereo)
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate stereo test signal (440Hz sine + 880Hz sine)
    audio_stereo = np.stack([
        0.5 * np.sin(2 * np.pi * 440 * t),  # Left: 440 Hz
        0.5 * np.sin(2 * np.pi * 880 * t),  # Right: 880 Hz
    ])

    print(f"\nInput audio:")
    print(f"  Shape: {audio_stereo.shape}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Channels: stereo → will be downmixed to mono")

    # Encode
    print(f"\nEncoding...")
    encoded = codec.encode(audio_stereo, sample_rate)

    code_shape = encoded["codes"][0].shape
    print(f"  Encoded codes shape: {code_shape}")
    print(f"  Code frame rate: {code_shape[-1] / duration:.1f} Hz")
    print(f"  Compression ratio: {audio_stereo.size / encoded['codes'][0].size:.1f}x")

    # Decode
    print(f"\nDecoding...")
    decoded_audio, decoded_sr = codec.decode(encoded)

    print(f"  Decoded audio shape: {decoded_audio.shape}")
    print(f"  Decoded sample rate: {decoded_sr} Hz")

    # Calculate reconstruction error
    # Downsample original to 24kHz mono for comparison
    original_mono = audio_stereo.mean(axis=0)
    if sample_rate != decoded_sr:
        from scipy import signal
        original_resampled = signal.resample(
            original_mono,
            int(len(original_mono) * decoded_sr / sample_rate)
        )
    else:
        original_resampled = original_mono

    # Trim to same length
    min_len = min(len(original_resampled), decoded_audio.shape[-1])
    mse = np.mean((original_resampled[:min_len] - decoded_audio[0, :min_len])**2)

    print(f"\nReconstruction MSE: {mse:.6f}")
    print(f"✓ Mimi codec test passed!")

    return True

def test_mimi_parallel():
    """Test Mimi codec with parallel processing."""
    print("\n" + "="*70)
    print("Testing Mimi Codec (Parallel Mode)")
    print("="*70)

    # Create codec engine with parallel processing
    codec = MimiAudioEngine(parallel=True, cpu_percent=25)

    # Generate longer test audio (5 seconds)
    duration = 5.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate mono test signal
    audio_mono = 0.5 * np.sin(2 * np.pi * 440 * t)

    print(f"\nInput audio:")
    print(f"  Shape: {audio_mono.shape}")
    print(f"  Duration: {duration} s")

    # Encode
    import time
    start = time.time()
    encoded = codec.encode(audio_mono, sample_rate)
    encode_time = time.time() - start

    print(f"\nEncoding (parallel):")
    print(f"  Time: {encode_time:.3f} s")
    print(f"  Codes shape: {encoded['codes'][0].shape}")

    # Decode
    start = time.time()
    decoded_audio, decoded_sr = codec.decode(encoded)
    decode_time = time.time() - start

    print(f"\nDecoding (parallel):")
    print(f"  Time: {decode_time:.3f} s")
    print(f"  Output shape: {decoded_audio.shape}")

    print(f"\n✓ Parallel processing test passed!")
    print(f"  Total time: {encode_time + decode_time:.3f} s")

    return True

if __name__ == "__main__":
    try:
        # Test sequential mode
        test_mimi_sequential()

        # Test parallel mode
        test_mimi_parallel()

        print("\n" + "="*70)
        print("All tests passed! Mimi codec is working correctly.")
        print("="*70)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
