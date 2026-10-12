"""Random baseline agent for sound classification.

Tests environment with random guessing. Expected accuracy: ~25% (chance level with 4 classes).
"""

from physonic.envs import SoundClassificationEnv

# Create environment with shorter audio for faster testing
env = SoundClassificationEnv(
    audio_duration=0.5,
    hrtf_path="assets/hrtf/test_kemar/"  # Use test HRTF dataset
)

correct = 0
total = 100

print("Running random baseline agent...")
print("=" * 60)

for episode in range(total):
    obs, info = env.reset()
    action = env.action_space.sample()  # Random guess
    obs, reward, done, trunc, info = env.step(action)

    if info["correct"]:
        correct += 1

    symbol = "✓" if info["correct"] else "✗"
    print(f"Episode {episode:3d}: {info['true_material']:6s} -> {info['predicted_material']:6s} ({symbol})")

print("=" * 60)
print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%")
print(f"Expected: ~25% (random chance with 4 classes)")

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
