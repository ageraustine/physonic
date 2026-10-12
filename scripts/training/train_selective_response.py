"""Train a simple PyTorch policy for selective response task.

This script demonstrates how to train a neural network policy using REINFORCE
(policy gradient) algorithm. The policy learns to classify sounds as targets
or distractors based on mel-spectrogram observations.

Requirements:
    - torch (install with: pip install torch)

Expected Results:
    - Random baseline: ~50% precision, ~50% recall
    - Trained policy: Should improve F1 score over random baseline
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from physonic.envs import SelectiveResponseEnv


class SimplePolicy(nn.Module):
    """Simple feedforward policy for mel-spectrogram input."""

    def __init__(self, input_shape=(64, 44, 2), hidden_size=128):
        super().__init__()

        # Flatten mel-spectrogram input
        input_size = np.prod(input_shape)

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),  # 2 actions: IGNORE, RESPOND
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """Forward pass: obs -> action probabilities."""
        return self.network(x)


def collect_episode(env, policy, device):
    """Collect one episode of experience."""
    obs, info = env.reset()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

    # Get action probabilities
    with torch.no_grad():
        action_probs = policy(obs_tensor)

    # Sample action
    action_dist = torch.distributions.Categorical(action_probs)
    action = action_dist.sample().item()
    log_prob = action_dist.log_prob(torch.tensor(action))

    # Take step
    next_obs, reward, done, trunc, info = env.step(action)

    return {
        "obs": obs,
        "action": action,
        "log_prob": log_prob,
        "reward": reward,
        "info": info
    }


def train_policy_gradient(env, policy, optimizer, num_episodes=1000):
    """Simple REINFORCE training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    print(f"Training on device: {device}")
    print("=" * 60)

    for episode in range(num_episodes):
        # Collect episode
        episode_data = collect_episode(env, policy, device)

        # Compute loss (REINFORCE: maximize expected reward)
        loss = -episode_data["log_prob"] * episode_data["reward"]

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress
        if episode % 100 == 0:
            metrics = env.get_metrics()
            print(f"Episode {episode:4d}: "
                  f"Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, "
                  f"F1={metrics['f1']:.3f}")
            env.reset_metrics()


def evaluate_policy(env, policy, num_episodes=100):
    """Evaluate trained policy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()

    env.reset_metrics()

    with torch.no_grad():
        for _ in range(num_episodes):
            obs, info = env.reset()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            # Get action (greedy)
            action_probs = policy(obs_tensor)
            action = action_probs.argmax().item()

            env.step(action)

    return env.get_metrics()


if __name__ == "__main__":
    # Check if PyTorch is available
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed. Install with: pip install torch")
        print("This script is optional - you can skip it if you don't need PyTorch training.")
        exit(1)

    # Create environment
    env = SelectiveResponseEnv(
        target_materials=["glass"],
        audio_duration=0.5,
        target_probability=0.3,
        hrtf_path="assets/hrtf/test_kemar/"
    )

    print("Selective Response PyTorch Training")
    print("=" * 60)
    print(f"Environment: {env.__class__.__name__}")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n} (IGNORE, RESPOND)")
    print(f"Target materials: {env.target_materials}")
    print(f"Distractor materials: {env.distractor_materials}")
    print(f"Target probability: {env.target_probability:.1%}")
    print()

    # Create policy
    policy = SimplePolicy(input_shape=env.observation_space.shape)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # Evaluate random baseline
    print("Random Baseline (before training):")
    baseline_metrics = evaluate_policy(env, policy, num_episodes=100)
    print(f"  Precision: {baseline_metrics['precision']:.3f}")
    print(f"  Recall:    {baseline_metrics['recall']:.3f}")
    print(f"  F1 Score:  {baseline_metrics['f1']:.3f}")
    print(f"  Accuracy:  {baseline_metrics['accuracy']:.3f}")
    print()

    # Train
    print("Training policy with REINFORCE...")
    train_policy_gradient(env, policy, optimizer, num_episodes=1000)
    print()

    # Final evaluation
    print("Final Evaluation (after training):")
    final_metrics = evaluate_policy(env, policy, num_episodes=100)
    print(f"  Precision: {final_metrics['precision']:.3f}")
    print(f"  Recall:    {final_metrics['recall']:.3f}")
    print(f"  F1 Score:  {final_metrics['f1']:.3f}")
    print(f"  Accuracy:  {final_metrics['accuracy']:.3f}")
    print()

    # Show confusion matrix
    cm = env.get_confusion_matrix()
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(f"  Negative: [{cm[0, 0]:3.0f} (TN), {cm[0, 1]:3.0f} (FP)]")
    print(f"  Positive: [{cm[1, 0]:3.0f} (FN), {cm[1, 1]:3.0f} (TP)]")
    print()

    # Compare to baseline
    print("Improvement over random baseline:")
    print(f"  ΔPrecision: {final_metrics['precision'] - baseline_metrics['precision']:+.3f}")
    print(f"  ΔRecall:    {final_metrics['recall'] - baseline_metrics['recall']:+.3f}")
    print(f"  ΔF1:        {final_metrics['f1'] - baseline_metrics['f1']:+.3f}")
    print(f"  ΔAccuracy:  {final_metrics['accuracy'] - baseline_metrics['accuracy']:+.3f}")
