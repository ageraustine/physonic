"""Never-respond baseline agent for selective response task.

Strategy: Always IGNORE every sound.
Expected: Precision undefined (no predictions), Recall = 0% (misses all targets).
"""

from physonic.envs import SelectiveResponseEnv

# Create environment with shorter audio for faster testing
env = SelectiveResponseEnv(
    target_materials=["glass"],
    audio_duration=0.5,
    target_probability=0.3,  # 30% targets
    hrtf_path="assets/hrtf/test_kemar/"
)

print("Running 'never respond' baseline agent...")
print("=" * 60)

for episode in range(100):
    obs, info = env.reset()
    action = 0  # Always IGNORE
    obs, reward, done, trunc, info = env.step(action)

    # Display symbol based on outcome
    symbols = {"TP": "✓", "TN": "✓", "FP": "✗", "FN": "✗"}
    symbol = symbols[info["outcome"]]
    target_str = "TARGET" if info["is_target"] else "DISTRACTOR"
    action_str = "RESPOND" if info["agent_responds"] else "IGNORE"

    print(f"Episode {episode:3d}: {target_str:10s} -> {action_str:7s} [{info['outcome']:2s}] ({symbol})")

print("=" * 60)

# Get final metrics
metrics = env.get_metrics()
print(f"\nMetrics:")
print(f"  Precision: {metrics['precision']:.3f}  (undefined - no predictions)")
print(f"  Recall:    {metrics['recall']:.3f}  (0% - misses all targets)")
print(f"  F1 Score:  {metrics['f1']:.3f}")
print(f"  Accuracy:  {metrics['accuracy']:.3f}")

# Show confusion matrix
cm = env.get_confusion_matrix()
print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
print(f"  Negative: [{cm[0, 0]:3.0f} (TN), {cm[0, 1]:3.0f} (FP)]")
print(f"  Positive: [{cm[1, 0]:3.0f} (FN), {cm[1, 1]:3.0f} (TP)]")

# Detailed counts
print(f"\nDetailed Counts:")
print(f"  True Positives (TP):  {metrics['tp']}")
print(f"  True Negatives (TN):  {metrics['tn']}")
print(f"  False Positives (FP): {metrics['fp']}")
print(f"  False Negatives (FN): {metrics['fn']}")

print(f"\nExpected: Recall = 0%, Accuracy ≈ 70% (distractor rate)")
