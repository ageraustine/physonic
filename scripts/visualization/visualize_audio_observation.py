"""Visualize how RL agents perceive audio observations.

This script demonstrates the difference between:
1. Raw audio waveform (what a microphone captures)
2. Mel-spectrogram (what the agent sees)
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from physonic.envs.data_center_nav import DataCenterNavigationEnv


def visualize_audio_perception():
    """Compare raw audio vs mel-spectrogram representations."""

    print("=" * 70)
    print("Audio Perception Visualization")
    print("=" * 70)

    # Create environments with both observation types
    print("\n1. Creating environment with mel-spectrogram observation...")
    env_mel = DataCenterNavigationEnv(
        use_mel_spectrogram=True,
        n_mels=64,
        audio_duration=0.5,
        seed=42
    )

    print("2. Creating environment with raw audio observation...")
    env_raw = DataCenterNavigationEnv(
        use_mel_spectrogram=False,
        audio_duration=0.5,
        seed=42
    )

    # Reset and get observations
    print("\n3. Resetting environments and capturing observations...")
    obs_mel, _ = env_mel.reset()
    obs_raw, _ = env_raw.reset()

    audio_mel = obs_mel["audio"]  # Shape: (64, 50, 2)
    audio_raw = obs_raw["audio"]  # Shape: (22050, 2)

    print(f"\nObservation shapes:")
    print(f"  Mel-spectrogram: {audio_mel.shape} (n_mels, time_frames, channels)")
    print(f"  Raw audio: {audio_raw.shape} (samples, channels)")
    print(f"  Sample rate: 44100 Hz")
    print(f"  Duration: 0.5 seconds")

    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10))

    # Row 1: Raw waveforms (left and right channels)
    ax1 = plt.subplot(4, 2, 1)
    time = np.linspace(0, 0.5, len(audio_raw))
    ax1.plot(time, audio_raw[:, 0], 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.set_title('Raw Audio - Left Ear (What Microphone Captures)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.5)
    ax1.set_ylim(-1, 1)

    ax2 = plt.subplot(4, 2, 2)
    ax2.plot(time, audio_raw[:, 1], 'r-', linewidth=0.5, alpha=0.7)
    ax2.set_ylabel('Amplitude', fontsize=10)
    ax2.set_title('Raw Audio - Right Ear (What Microphone Captures)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(-1, 1)

    # Row 2: Mel-spectrograms (what agent sees)
    ax3 = plt.subplot(4, 2, 3)
    mel_left = audio_mel[:, :, 0]  # (64, 50)
    im3 = ax3.imshow(mel_left, aspect='auto', origin='lower', cmap='magma',
                     vmin=-80, vmax=0, interpolation='nearest')
    ax3.set_ylabel('Mel Frequency Bin', fontsize=10)
    ax3.set_title('Mel-Spectrogram - Left Ear (What Agent Sees)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Time Frame (~10ms per frame)', fontsize=9)
    plt.colorbar(im3, ax=ax3, label='Power (dB)')

    ax4 = plt.subplot(4, 2, 4)
    mel_right = audio_mel[:, :, 1]  # (64, 50)
    im4 = ax4.imshow(mel_right, aspect='auto', origin='lower', cmap='magma',
                     vmin=-80, vmax=0, interpolation='nearest')
    ax4.set_ylabel('Mel Frequency Bin', fontsize=10)
    ax4.set_title('Mel-Spectrogram - Right Ear (What Agent Sees)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time Frame (~10ms per frame)', fontsize=9)
    plt.colorbar(im4, ax=ax4, label='Power (dB)')

    # Row 3: Zoomed-in waveform segment
    ax5 = plt.subplot(4, 2, 5)
    zoom_start = 0
    zoom_end = 1000  # First 1000 samples (~23ms)
    time_zoom = np.linspace(0, zoom_end/44100, zoom_end)
    ax5.plot(time_zoom * 1000, audio_raw[zoom_start:zoom_end, 0], 'b-', linewidth=1)
    ax5.set_xlabel('Time (ms)', fontsize=9)
    ax5.set_ylabel('Amplitude', fontsize=10)
    ax5.set_title('Zoomed Raw Audio (First 23ms)', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Row 3: Single mel-spectrogram column (what agent sees at one moment)
    ax6 = plt.subplot(4, 2, 6)
    mel_column = audio_mel[:, 25, :]  # Middle time frame, both channels
    ax6.plot(mel_column[:, 0], range(64), 'b-', linewidth=2, label='Left ear')
    ax6.plot(mel_column[:, 1], range(64), 'r-', linewidth=2, alpha=0.7, label='Right ear')
    ax6.set_xlabel('Power (dB)', fontsize=9)
    ax6.set_ylabel('Mel Frequency Bin', fontsize=10)
    ax6.set_title('Single Time Frame (What Agent Sees at t=0.25s)', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=9)
    ax6.set_xlim(-80, 0)

    # Row 4: Information comparison
    ax7 = plt.subplot(4, 2, 7)
    ax7.axis('off')
    info_raw = f"""
    RAW AUDIO (Waveform)

    Format: Time-series signal
    Shape: (22,050, 2)
    - 22,050 samples (0.5s @ 44.1kHz)
    - 2 channels (binaural)

    Advantages:
    ✓ Complete frequency information
    ✓ Preserves all details
    ✓ No information loss

    Disadvantages:
    ✗ Too much data for neural networks
    ✗ 44,100 input features per second
    ✗ Requires very deep networks
    ✗ Hard to learn temporal patterns

    Use case: Traditional signal processing
    """
    ax7.text(0.1, 0.5, info_raw, fontsize=9, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')
    info_mel = f"""
    MEL-SPECTROGRAM (Image)

    Format: 2D frequency-time representation
    Shape: (64, 50, 2)
    - 64 mel frequency bins
    - 50 time frames (~10ms per frame)
    - 2 channels (binaural)

    Advantages:
    ✓ Compact representation (3,200 vs 22,050)
    ✓ Works well with CNNs
    ✓ Mimics human hearing
    ✓ Captures important features

    Disadvantages:
    ✗ Some frequency resolution lost
    ✗ Phase information lost

    Use case: Deep learning (THIS IS WHAT THE AGENT USES!)
    Total input: 6,400 features (audio) + 6 (state) = 6,406
    """
    ax8.text(0.1, 0.5, info_mel, fontsize=9, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    plt.savefig('outputs/audio_perception_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: outputs/audio_perception_visualization.png")

    # Create neural network architecture diagram
    fig2, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    network_diagram = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    PPO AGENT ARCHITECTURE                            │
    │                                                                       │
    │  OBSERVATION (Dict Space)                                            │
    │  ┌─────────────────────┐     ┌──────────────────┐                  │
    │  │ Audio (Mel-Spec)    │     │ State (Robot)    │                  │
    │  │ (64, 50, 2)         │     │ (6,)             │                  │
    │  │                     │     │                  │                  │
    │  │ 64 mel bins         │     │ [x, y, z,        │                  │
    │  │ 50 time frames      │     │  vx, vy, heading]│                  │
    │  │ 2 channels          │     │                  │                  │
    │  └──────────┬──────────┘     └────────┬─────────┘                  │
    │             │                         │                             │
    │             │ Flatten                 │ Identity                    │
    │             ▼                         ▼                             │
    │       ┌──────────┐             ┌──────────┐                        │
    │       │  6,400   │             │    6     │                        │
    │       └─────┬────┘             └────┬─────┘                        │
    │             │                       │                               │
    │             └───────────┬───────────┘                               │
    │                         │ Concatenate                               │
    │                         ▼                                           │
    │                  ┌─────────────┐                                    │
    │                  │    6,406    │  Combined Feature Vector           │
    │                  └──────┬──────┘                                    │
    │                         │                                           │
    │         ┌───────────────┴───────────────┐                           │
    │         │                               │                           │
    │         ▼                               ▼                           │
    │  ┌─────────────┐                 ┌─────────────┐                   │
    │  │ ACTOR (π)   │                 │ CRITIC (V)  │                   │
    │  │             │                 │             │                   │
    │  │ FC 6406→256 │                 │ FC 6406→256 │                   │
    │  │ ReLU        │                 │ ReLU        │                   │
    │  │ FC 256→256  │                 │ FC 256→256  │                   │
    │  │ ReLU        │                 │ ReLU        │                   │
    │  │ FC 256→2    │                 │ FC 256→1    │                   │
    │  └──────┬──────┘                 └──────┬──────┘                   │
    │         │                               │                           │
    │         ▼                               ▼                           │
    │   ┌──────────┐                   ┌──────────┐                      │
    │   │ Action   │                   │  Value   │                      │
    │   │ [vx, vy] │                   │ Estimate │                      │
    │   └──────────┘                   └──────────┘                      │
    │                                                                     │
    │  Total Parameters: ~1.6M (mostly in first layer 6406→256)          │
    └─────────────────────────────────────────────────────────────────────┘

    Key Points:
    • Audio is flattened: 64×50×2 = 6,400 features
    • State is small: just 6 features (position + velocity + heading)
    • Agent learns to map (mel-spectrogram + state) → actions
    • CNN could be added for audio feature extraction (future work)
    • Currently uses simple MLP with flat audio
    """

    ax.text(0.5, 0.5, network_diagram, fontsize=10, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.2))

    plt.tight_layout()
    plt.savefig('outputs/ppo_architecture_diagram.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved architecture diagram to: outputs/ppo_architecture_diagram.png")

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"✓ Raw audio: {audio_raw.shape[0]:,} samples × 2 channels = {audio_raw.size:,} numbers")
    print(f"✓ Mel-spectrogram: 64 bins × 50 frames × 2 channels = {audio_mel.size:,} numbers")
    print(f"✓ Compression ratio: {audio_raw.size / audio_mel.size:.1f}x smaller")
    print(f"\nThe agent sees audio as a 2D image (like a photograph of sound)")
    print(f"where X-axis = time, Y-axis = frequency, and color = intensity.")
    print(f"\nThis is similar to how image classification works, but for sound!")


if __name__ == "__main__":
    visualize_audio_perception()
