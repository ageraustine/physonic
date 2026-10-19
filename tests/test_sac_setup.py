"""Test that SAC training setup works correctly."""

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from physonic.envs.data_center_nav import DataCenterNavigationEnv


def test_sac_setup():
    """Test SAC with data center navigation environment."""
    print("=" * 70)
    print("SAC Setup Test")
    print("=" * 70)
    print()

    # Create environment
    print("Creating environment...")
    def make_env():
        env = DataCenterNavigationEnv(
            room_size=(30, 25, 4),
            num_racks=24,
            use_goal_sound=True,
            reward_shaping="progress",
            seed=42,
        )
        return Monitor(env)

    env = DummyVecEnv([make_env])
    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print()

    # Create SAC model
    print("Creating SAC model...")
    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=10000,  # Small buffer for testing
        learning_starts=100,  # Quick start for testing
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",  # Automatic entropy tuning
        verbose=1,
        device="cpu",
    )
    print(f"✓ SAC model created")
    print(f"  Policy: {model.policy}")
    print(f"  Learning rate: {model.learning_rate}")
    print(f"  Buffer size: {model.buffer_size}")
    print(f"  Entropy coefficient: {model.ent_coef}")
    print()

    # Test initial observation
    print("Testing observation...")
    obs = env.reset()
    print(f"✓ Observation shape: {obs['audio_codes'].shape}, {obs['ultrasonic'].shape}, {obs['state'].shape}")
    print()

    # Test action prediction
    print("Testing action prediction...")
    action, _states = model.predict(obs, deterministic=False)
    print(f"✓ Action: {action}")
    print(f"  Forward speed: {action[0][0]:.3f}")
    print(f"  Angular velocity: {action[0][1]:.3f}")
    print()

    # Test environment step
    print("Testing environment step...")
    obs, reward, done, info = env.step(action)
    print(f"✓ Step completed")
    print(f"  Reward: {reward[0]:.3f}")
    print(f"  Done: {done[0]}")
    print(f"  Distance to goal: {info[0]['distance_to_goal']:.2f}m")
    print()

    # Test short training run
    print("Testing short training run (1000 steps)...")
    model.learn(total_timesteps=1000, log_interval=10, progress_bar=False)
    print(f"✓ Training completed")
    print(f"  Replay buffer size: {model.replay_buffer.size()}")
    print()

    # Test trained model prediction
    print("Testing trained model prediction...")
    obs = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    print(f"✓ Prediction after training: {action}")
    print()

    print("=" * 70)
    print("SAC Setup Test Complete!")
    print("=" * 70)
    print()
    print("✅ SAC is ready for full training")
    print()
    print("To train SAC on full task:")
    print("  PYTHONPATH=. python3 scripts/training/train_sac_data_center.py \\")
    print("    --total-timesteps 500000 --buffer-size 100000")
    print()
    print("Key SAC advantages for audio navigation:")
    print("  • Maximum entropy exploration → discovers binaural cues")
    print("  • Experience replay → 3-5x more sample efficient")
    print("  • Off-policy learning → reuses expensive audio observations")
    print()

    env.close()


if __name__ == "__main__":
    test_sac_setup()
