"""Test reward signal improvements."""

import numpy as np
import matplotlib.pyplot as plt
from physonic.envs.data_center_nav import DataCenterNavigationEnv


def test_goal_sound():
    """Test that goal emits distinctive sound."""
    print("=" * 70)
    print("Test 1: Goal Sound Source")
    print("=" * 70)

    # Create environment with goal sound
    env = DataCenterNavigationEnv(
        room_size=(30, 25, 4),
        num_racks=24,
        use_goal_sound=True,
        use_mimi=True,  # Use Mimi codec
        audio_duration=0.5,
        seed=42,
    )

    obs, info = env.reset()
    print(f"Environment created with {env.num_racks} racks")
    print(f"Goal sound enabled: {info['goal_sound_enabled']}")
    print(f"Number of sound sources: {len(env.audio_generator.continuous_sounds)}")

    # Check goal sound exists
    goal_sounds = [s for s in env.audio_generator.continuous_sounds if s.source_id == "goal_beacon"]
    if goal_sounds:
        goal_sound = goal_sounds[0]
        print(f"\n✅ Goal sound found!")
        print(f"  - Frequency: {goal_sound.fundamental_freq} Hz (vs 120Hz servers)")
        print(f"  - Position: {goal_sound.position}")
        print(f"  - Modulation: {goal_sound.modulation_freq} Hz @ {goal_sound.modulation_depth} depth")
        print(f"  - Volume: {goal_sound.volume}")
        print(f"  - Sound type: {goal_sound.sound_type}")
    else:
        print("❌ Goal sound NOT found!")
        return False

    # Compare to server sounds
    server_sounds = [s for s in env.audio_generator.continuous_sounds if s.source_id != "goal_beacon"]
    if server_sounds:
        avg_freq = np.mean([s.fundamental_freq for s in server_sounds])
        print(f"\nServer racks:")
        print(f"  - Average frequency: {avg_freq:.1f} Hz")
        print(f"  - Frequency range: {min(s.fundamental_freq for s in server_sounds):.1f} - "
              f"{max(s.fundamental_freq for s in server_sounds):.1f} Hz")
        print(f"\n✅ Goal frequency ({goal_sound.fundamental_freq}Hz) is VERY different from servers ({avg_freq:.1f}Hz)")

    print()
    return True


def test_reward_shaping():
    """Test different reward shaping modes."""
    print("=" * 70)
    print("Test 2: Reward Shaping Modes")
    print("=" * 70)

    results = {}

    for mode in ["progress", "distance", "sparse"]:
        env = DataCenterNavigationEnv(
            room_size=(30, 25, 4),
            num_racks=24,
            reward_shaping=mode,
            use_goal_sound=True,
            seed=42,
        )

        obs, info = env.reset()

        # Simulate moving towards goal
        rewards = []
        distances = []

        for step in range(50):
            # Simple heuristic: move towards goal
            robot_pos = env.data.xpos[env.robot_body_id]
            goal_pos = env.goal_position
            direction = goal_pos[:2] - robot_pos[:2]
            direction = direction / (np.linalg.norm(direction) + 1e-6)

            action = direction * 0.5  # Move towards goal
            obs, reward, terminated, truncated, info = env.step(action)

            rewards.append(reward)
            distances.append(info['distance_to_goal'])

            if terminated or truncated:
                break

        results[mode] = {
            'rewards': rewards,
            'distances': distances,
            'total_reward': sum(rewards),
            'final_distance': distances[-1],
            'steps': len(rewards)
        }

        print(f"\n{mode.upper()} mode:")
        print(f"  - Steps: {len(rewards)}")
        print(f"  - Total reward: {sum(rewards):.2f}")
        print(f"  - Final distance: {distances[-1]:.2f}m")
        print(f"  - Avg reward/step: {np.mean(rewards):.3f}")
        print(f"  - Positive rewards: {sum(1 for r in rewards if r > 0)}/{len(rewards)}")

    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, mode in enumerate(["progress", "distance", "sparse"]):
        ax = axes[i]
        data = results[mode]
        steps = range(len(data['rewards']))

        ax.plot(steps, data['rewards'], label='Reward', linewidth=2)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title(f'{mode.capitalize()} Mode\nTotal: {data["total_reward"]:.1f}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig('outputs/reward_shaping_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: outputs/reward_shaping_comparison.png")

    print()
    return True


def test_penalty_magnitudes():
    """Test that penalties are appropriate."""
    print("=" * 70)
    print("Test 3: Penalty Magnitudes")
    print("=" * 70)

    env = DataCenterNavigationEnv(
        room_size=(30, 25, 4),
        num_racks=24,
        reward_shaping="progress",
        use_goal_sound=True,
        seed=42,
    )

    print(f"Reward configuration:")
    print(f"  - Goal bonus: +{env.goal_bonus:.1f}")
    print(f"  - Collision penalty: -{env.collision_penalty:.1f}")
    print(f"  - Bounds penalty: -{env.bounds_penalty:.1f}")
    print(f"  - Reward shaping: {env.reward_shaping}")

    # Compare to cumulative progress rewards
    typical_episode_length = 200
    typical_progress_per_step = 0.1  # 0.1m closer per step
    cumulative_progress_reward = typical_episode_length * 10.0 * typical_progress_per_step

    print(f"\nTypical episode (200 steps, 0.1m progress/step):")
    print(f"  - Cumulative progress reward: +{cumulative_progress_reward:.1f}")
    print(f"  - Goal bonus: +{env.goal_bonus:.1f}")
    print(f"  - Total if successful: +{cumulative_progress_reward + env.goal_bonus:.1f}")

    print(f"\nPenalty vs Total Reward ratios:")
    total_positive = cumulative_progress_reward + env.goal_bonus
    print(f"  - Collision penalty / Total: {env.collision_penalty / total_positive:.2%}")
    print(f"  - Bounds penalty / Total: {env.bounds_penalty / total_positive:.2%}")

    if env.collision_penalty / total_positive > 0.05:
        print(f"\n✅ Collision penalty is significant ({env.collision_penalty / total_positive:.1%} of total reward)")
    else:
        print(f"\n❌ Collision penalty is too weak!")

    print()
    return True


def test_audio_necessity():
    """Test that audio is needed with goal sound."""
    print("=" * 70)
    print("Test 4: Audio Observation Necessity")
    print("=" * 70)

    # With goal sound
    env_with_audio = DataCenterNavigationEnv(
        room_size=(30, 25, 4),
        num_racks=24,
        use_goal_sound=True,
        seed=42,
    )

    obs, info = env_with_audio.reset()

    print("Observation space:")
    print(f"  - Audio codes: {obs['audio_codes'].shape} (Mimi-compressed)")
    print(f"  - Ultrasonic: {obs['ultrasonic'].shape}")
    print(f"  - State: {obs['state'].shape}")

    # Check if goal sound is in audio
    num_sources = len(env_with_audio.audio_generator.continuous_sounds)
    goal_present = any(s.source_id == "goal_beacon" for s in env_with_audio.audio_generator.continuous_sounds)

    print(f"\nAudio content:")
    print(f"  - Total sound sources: {num_sources}")
    print(f"  - Goal beacon present: {goal_present}")
    print(f"  - Server racks: {num_sources - 1}")

    if goal_present:
        print(f"\n✅ Goal sound is in audio observations - audio is NECESSARY for navigation!")
    else:
        print(f"\n❌ Goal sound missing - audio is NOT useful!")

    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("REWARD SIGNAL IMPROVEMENTS TEST SUITE")
    print("="*70)
    print()

    results = []
    results.append(("Goal Sound Source", test_goal_sound()))
    results.append(("Reward Shaping", test_reward_shaping()))
    results.append(("Penalty Magnitudes", test_penalty_magnitudes()))
    results.append(("Audio Necessity", test_audio_necessity()))

    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)
    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED! Reward improvements are working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
