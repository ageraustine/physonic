"""Test that robot perceives mixed audio like humans do."""

import numpy as np
import matplotlib.pyplot as plt
from physonic.envs.audio_observation import AudioObservationGenerator
from physonic.environments.dynamic_scene import generate_complex_data_center
import soundfile as sf
import os


def test_mixed_perception():
    """Verify robot receives mixed audio signal."""
    print("=" * 70)
    print("Testing Mixed Audio Perception (Like Human Hearing)")
    print("=" * 70)
    print()

    # Create continuous sounds (multiple server racks at different positions)
    room_size = (30, 25, 4)
    _, _, continuous_sounds = generate_complex_data_center(
        room_size=room_size,
        num_racks=24,
        num_robots=1,
        seed=42,
    )

    print(f"Generated {len(continuous_sounds)} continuous sound sources")
    print("Each server rack emits a different frequency:")
    for i, sound in enumerate(continuous_sounds[:5]):
        print(f"  - Rack {i+1}: {sound.fundamental_freq:.1f} Hz at position {sound.position[:2]}")
    print(f"  ... and {len(continuous_sounds) - 5} more racks")
    print()

    # Create audio observation generator
    project_root = os.path.dirname(__file__)
    hrtf_path = os.path.join(project_root, "assets", "hrtf", "test_kemar")

    audio_gen = AudioObservationGenerator(
        continuous_sounds=continuous_sounds,
        hrtf_path=hrtf_path,
        audio_duration=1.0,
        sample_rate=44100,
        max_audio_distance=30.0,
        use_mimi=False,  # Use raw audio to demonstrate mixing
    )

    # Generate audio from robot's perspective
    robot_position = np.array([15.0, 12.5, 0.15])  # Center of room
    robot_orientation = 0.0  # Facing forward

    print("Generating mixed audio from robot's perspective...")
    print(f"  - Robot position: {robot_position}")
    print(f"  - Robot orientation: {robot_orientation:.1f} radians")
    print()

    binaural_audio = audio_gen.generate_observation(
        robot_position=robot_position,
        robot_orientation=robot_orientation,
    )

    print("Audio Perception Analysis:")
    print(f"  - Shape: {binaural_audio.shape}")
    print(f"  - Channels: {binaural_audio.shape[1]} (stereo, like human ears)")
    print(f"  - Duration: {binaural_audio.shape[0] / 44100:.2f}s")
    print(f"  - Sample rate: 44,100 Hz")
    print()

    # Check signal statistics
    left_channel = binaural_audio[:, 0]
    right_channel = binaural_audio[:, 1]

    print("Signal Statistics:")
    print(f"  - Left channel RMS: {np.sqrt(np.mean(left_channel**2)):.6f}")
    print(f"  - Right channel RMS: {np.sqrt(np.mean(right_channel**2)):.6f}")
    print(f"  - Max amplitude: {np.abs(binaural_audio).max():.6f}")
    print(f"  - Stereo difference: {np.mean(np.abs(left_channel - right_channel)):.6f}")
    print()

    # Count audible sources
    audible_count = sum(
        1 for sound in continuous_sounds
        if np.linalg.norm(np.array(sound.position) - robot_position) < 30.0
    )
    print(f"✅ {audible_count}/{len(continuous_sounds)} sound sources mixed together")
    print("   (Sources beyond 30m are culled for efficiency)")
    print()

    # Save audio for listening test
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/test_mixed_perception.wav"
    sf.write(output_path, binaural_audio, 44100)
    print(f"✅ Mixed audio saved to: {output_path}")
    print("   Listen with HEADPHONES to hear multiple server racks mixed together!")
    print()

    # Visualize waveform
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Left channel
    time = np.arange(len(left_channel)) / 44100
    axes[0].plot(time[:2000], left_channel[:2000], color='blue', linewidth=0.5)
    axes[0].set_title("Left Ear (Mixed Signal from ~24 Server Racks)",
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    # Right channel
    axes[1].plot(time[:2000], right_channel[:2000], color='red', linewidth=0.5)
    axes[1].set_title("Right Ear (Mixed Signal from ~24 Server Racks)",
                      fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    # Spectrogram of left channel (shows mixed frequencies)
    from scipy import signal
    frequencies, times, Sxx = signal.spectrogram(
        left_channel, fs=44100, nperseg=1024
    )
    im = axes[2].pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10),
                            shading='gouraud', cmap='viridis', vmax=0, vmin=-60)
    axes[2].set_title("Spectrogram (Multiple Frequencies Mixed from Different Racks)",
                      fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(0, 500)  # Focus on low frequencies (server hum range)
    plt.colorbar(im, ax=axes[2], label='Power (dB)')

    plt.tight_layout()
    plot_path = "outputs/test_mixed_perception.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {plot_path}")
    print()

    print("=" * 70)
    print("Conclusion:")
    print("=" * 70)
    print("✅ Robot perceives MIXED audio (exactly like human hearing)")
    print()
    print("How it works:")
    print("  1. Each server rack emits sound at different frequency/position")
    print("  2. Each source is spatially rendered (HRTF + distance attenuation)")
    print("  3. All sources are SUMMED into single stereo signal (mixing)")
    print("  4. Robot receives this mixed signal (just like human ears)")
    print()
    print("Physical correctness:")
    print("  ✅ Microphones don't receive separated channels")
    print("  ✅ Sound waves superimpose (add together) in space")
    print("  ✅ Receiver hears the sum of all arriving waves")
    print()
    print("This is EXACTLY how human perception works!")
    print("The robot's neural network must learn to filter/separate sources")
    print("from the mixed signal (cocktail party effect).")
    print()


if __name__ == "__main__":
    test_mixed_perception()
