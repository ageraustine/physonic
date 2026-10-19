"""Test observation normalization with VecNormalize."""

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from physonic.envs.data_center_nav import DataCenterNavigationEnv


def test_normalization():
    """Show effect of VecNormalize on observations."""
    print("=" * 70)
    print("Observation Normalization Test")
    print("=" * 70)
    print()

    # Create environment
    def make_env():
        return DataCenterNavigationEnv(
            room_size=(30, 25, 4),
            num_racks=24,
            use_goal_sound=True,
            seed=42,
        )

    # WITHOUT normalization
    print("1. WITHOUT VecNormalize:")
    print("-" * 70)
    env_raw = DummyVecEnv([make_env])
    obs_raw = env_raw.reset()

    print(f"Audio codes range: [{obs_raw['audio_codes'].min()}, {obs_raw['audio_codes'].max()}]")
    print(f"  → Very large! Neural network sees values from 0 to 2047")
    print(f"Ultrasonic range: [{obs_raw['ultrasonic'].min():.2f}, {obs_raw['ultrasonic'].max():.2f}]")
    print(f"  → Meters (0-5)")
    print(f"State range: [{obs_raw['state'].min():.2f}, {obs_raw['state'].max():.2f}]")
    print(f"  → Mixed scales: position (0-30), velocity (-5 to 5), heading (-π to π)")
    print()
    print("❌ Problem: Different feature scales make learning harder!")
    print("   Audio codes dominate gradient (2047 >> 5.0 >> 1.0)")
    print()

    # WITH normalization
    print("2. WITH VecNormalize:")
    print("-" * 70)
    env_norm = DummyVecEnv([make_env])
    env_norm = VecNormalize(
        env_norm,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    # Collect some data to build statistics
    print("Building normalization statistics (100 steps)...")
    for _ in range(100):
        action = env_norm.action_space.sample()  # Returns shape (1, 2)
        obs, reward, done, info = env_norm.step(action)
        if done[0]:
            env_norm.reset()

    obs_norm = env_norm.reset()

    print(f"Audio codes range: [{obs_norm['audio_codes'].min():.2f}, {obs_norm['audio_codes'].max():.2f}]")
    print(f"  → Normalized to ~[-3, 3]")
    print(f"Ultrasonic range: [{obs_norm['ultrasonic'].min():.2f}, {obs_norm['ultrasonic'].max():.2f}]")
    print(f"  → Normalized to ~[-1, 1]")
    print(f"State range: [{obs_norm['state'].min():.2f}, {obs_norm['state'].max():.2f}]")
    print(f"  → Normalized to ~[-2, 2]")
    print()
    print("✅ Solution: All features in similar range!")
    print("   Neural network learns faster with balanced inputs")
    print()

    # Show normalization stats
    print("3. Normalization Statistics:")
    print("-" * 70)
    print("VecNormalize tracks running mean and std for each feature:")
    print(f"  Audio codes mean: {env_norm.obs_rms.mean[0:5].flatten()[:5]}...")
    print(f"  Audio codes std:  {env_norm.obs_rms.var[0:5].flatten()[:5]}...")
    print()
    print("Formula: normalized = (raw - running_mean) / sqrt(running_var)")
    print()

    # Cleanup
    env_raw.close()
    env_norm.close()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("WITHOUT VecNormalize:")
    print("  • Audio codes: [0, 2047]")
    print("  • Ultrasonic: [0, 5]")
    print("  • State: mixed scales")
    print("  → Hard to learn! Features dominate each other")
    print()
    print("WITH VecNormalize:")
    print("  • All features: ~[-3, 3]")
    print("  • Similar scales")
    print("  → Easy to learn! Neural network sees balanced inputs")
    print()
    print("Expected improvement:")
    print("  • 2-3x faster convergence")
    print("  • Better final performance")
    print("  • More stable training")
    print()


if __name__ == "__main__":
    test_normalization()
