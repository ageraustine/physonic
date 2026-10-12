"""Frequency-based heuristic baseline for sound classification.

Uses spectral centroid (brightness) to classify materials.
Expected accuracy: ~40-60% (simple frequency-based heuristic).

Material characteristics:
- Steel:  f0=1400 Hz, bright, long decay
- Glass:  f0=2200 Hz, very bright, moderate decay
- Wood:   f0=600 Hz, warm, short decay
- Rubber: f0=180 Hz, dull, very short decay
"""

import numpy as np
from physonic.envs import SoundClassificationEnv


def simple_classifier(obs):
    """Simple heuristic based on mel-spectrogram energy distribution.

    Args:
        obs: Mel-spectrogram (n_mels, time_frames, 2) in dB scale

    Returns:
        Predicted class index (0=steel, 1=glass, 2=wood, 3=rubber)
    """
    # Average over time and stereo channels
    mel_avg = obs.mean(axis=(1, 2))  # (64,) average mel spectrum (dB)

    # Convert dB to linear scale for energy computation
    mel_linear = 10 ** (mel_avg / 20.0)  # dB to amplitude

    # Compute energy in frequency bands
    low_energy = mel_linear[:20].sum()     # Low freqs (20-700 Hz)
    mid_energy = mel_linear[20:45].sum()   # Mid freqs (700-3500 Hz)
    high_energy = mel_linear[45:].sum()    # High freqs (3500-8000 Hz)

    # Compute spectral centroid (brightness) - center of mass of spectrum
    freqs = np.linspace(20, 8000, 64)
    total_energy = mel_linear.sum() + 1e-10
    centroid = np.sum(freqs * mel_linear) / total_energy

    # Energy ratio: high-to-low ratio indicates brightness
    brightness_ratio = (high_energy + 0.1) / (low_energy + 0.1)

    # Decision tree based on centroid and brightness
    if brightness_ratio > 2.0 and centroid > 2500:
        return 1  # glass (very bright, high f0, high brightness ratio)
    elif brightness_ratio > 1.0 and centroid > 1200:
        return 0  # steel (bright, mid-high f0)
    elif centroid > 400:
        return 2  # wood (warm, lower f0)
    else:
        return 3  # rubber (dull, lowest f0, low centroid)


# Create environment with shorter audio for faster testing
env = SoundClassificationEnv(
    audio_duration=0.5,
    use_mel_spectrogram=True,
    hrtf_path="assets/hrtf/test_kemar/"  # Use test HRTF dataset
)

correct = 0
total = 100

print("Running frequency-based heuristic baseline...")
print("=" * 60)

for episode in range(total):
    obs, info = env.reset()
    action = simple_classifier(obs)
    obs, reward, done, trunc, info = env.step(action)

    if info["correct"]:
        correct += 1

    symbol = "✓" if info["correct"] else "✗"
    print(f"Episode {episode:3d}: {info['true_material']:6s} -> {info['predicted_material']:6s} ({symbol})")

print("=" * 60)
print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%")
print(f"Expected: ~40-60% (simple frequency-based heuristic)")

# Show confusion matrix
cm = env.get_confusion_matrix()
print("\nConfusion Matrix (rows=true, cols=predicted):")
print("        steel  glass  wood  rubber")
for i, material in enumerate(env.MATERIAL_CLASSES):
    print(f"{material:6s}  {cm[i, 0]:5.0f}  {cm[i, 1]:5.0f}  {cm[i, 2]:4.0f}  {cm[i, 3]:6.0f}")

# Per-class accuracy
print("\nPer-class Accuracy:")
for i, material in enumerate(env.MATERIAL_CLASSES):
    if cm[i].sum() > 0:
        acc = cm[i, i] / cm[i].sum()
        print(f"  {material:6s}: {100*acc:.1f}%")
