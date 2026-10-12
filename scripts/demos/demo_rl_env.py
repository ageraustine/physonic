"""Interactive demo for Physonic RL environments.

This script demonstrates the three RL environments with visualization:
1. Sound Localization (continuous action - predict azimuth)
2. Sound Classification (discrete action - classify material)
3. Selective Response (binary action - respond/ignore)

Compatible with:
- Gymnasium (standard RL interface)
- HuggingFace (can export episodes)
- Isaac Sim (future USD export)

Usage:
    python3 demo_rl_env.py --env localization --episodes 5
    python3 demo_rl_env.py --env classification --episodes 10 --visualize
    python3 demo_rl_env.py --env selective --agent perfect
"""

import argparse
import numpy as np
from pathlib import Path

from physonic.envs import (
    SoundLocalizationEnv,
    SoundClassificationEnv,
    SelectiveResponseEnv,
)
from physonic.visualization.recorder import EpisodeRecorder
from physonic.visualization.plots import (
    plot_mel_spectrogram,
    plot_metrics,
    plot_confusion_matrix,
    plot_episode_summary,
)


def run_random_agent(env, num_episodes: int, recorder: EpisodeRecorder):
    """Run random agent on environment.

    Args:
        env: Gymnasium environment
        num_episodes: Number of episodes to run
        recorder: Episode recorder

    Returns:
        List of episode rewards
    """
    episode_rewards = []

    for episode in range(num_episodes):
        recorder.start_episode(env, metadata={"agent": "random", "episode": episode})

        obs, info = env.reset(seed=episode)
        episode_reward = 0
        done = False
        step = 0

        while not done and step < 100:
            action = env.action_space.sample()
            recorder.record_step(obs, action, 0.0, info)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1

            recorder.record_step(obs, action, reward, info)

        recorder.save_episode()
        episode_rewards.append(episode_reward)

        print(f"Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.3f}")

    return episode_rewards


def run_perfect_agent_localization(env, num_episodes: int, recorder: EpisodeRecorder):
    """Run perfect agent on sound localization (cheats using ground truth).

    Args:
        env: SoundLocalizationEnv
        num_episodes: Number of episodes
        recorder: Episode recorder

    Returns:
        List of episode rewards
    """
    episode_rewards = []

    for episode in range(num_episodes):
        recorder.start_episode(env, metadata={"agent": "perfect", "episode": episode})

        obs, info = env.reset(seed=episode)
        episode_reward = 0

        # Perfect agent predicts true azimuth
        action = [info["true_azimuth"]]

        recorder.record_step(obs, action, 0.0, info)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        recorder.record_step(obs, action, reward, info)
        recorder.save_episode()
        episode_rewards.append(episode_reward)

        print(f"Episode {episode + 1}/{num_episodes}: Reward = {reward:.3f} "
              f"(Error: {info['angular_error']:.1f}°)")

    return episode_rewards


def run_perfect_agent_classification(env, num_episodes: int, recorder: EpisodeRecorder):
    """Run perfect agent on classification (cheats using ground truth).

    Args:
        env: SoundClassificationEnv
        num_episodes: Number of episodes
        recorder: Episode recorder

    Returns:
        List of episode rewards
    """
    episode_rewards = []

    for episode in range(num_episodes):
        recorder.start_episode(env, metadata={"agent": "perfect", "episode": episode})

        obs, info = env.reset(seed=episode)
        episode_reward = 0

        # Perfect agent predicts true material
        material_to_action = {mat: i for i, mat in enumerate(env.MATERIAL_CLASSES)}
        action = material_to_action[info["true_material"]]

        recorder.record_step(obs, action, 0.0, info)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        recorder.record_step(obs, action, reward, info)
        recorder.save_episode()
        episode_rewards.append(episode_reward)

        correct_str = "✓" if info["correct"] else "✗"
        print(f"Episode {episode + 1}/{num_episodes}: {correct_str} "
              f"True: {info['true_material']}, Pred: {info['predicted_material']}")

    return episode_rewards


def run_perfect_agent_selective(env, num_episodes: int, recorder: EpisodeRecorder):
    """Run perfect agent on selective response (cheats using ground truth).

    Args:
        env: SelectiveResponseEnv
        num_episodes: Number of episodes
        recorder: Episode recorder

    Returns:
        List of episode rewards
    """
    episode_rewards = []

    for episode in range(num_episodes):
        recorder.start_episode(env, metadata={"agent": "perfect", "episode": episode})

        obs, info = env.reset(seed=episode)
        episode_reward = 0

        # Perfect agent responds to targets, ignores distractors
        action = 1 if info["is_target"] else 0

        recorder.record_step(obs, action, 0.0, info)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        recorder.record_step(obs, action, reward, info)
        recorder.save_episode()
        episode_rewards.append(episode_reward)

        print(f"Episode {episode + 1}/{num_episodes}: {info['outcome']} "
              f"(Material: {info['material']}, Target: {info['is_target']})")

    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Physonic RL Environment Demo")
    parser.add_argument(
        "--env",
        type=str,
        choices=["localization", "classification", "selective"],
        default="localization",
        help="Environment to demo",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["random", "perfect"],
        default="random",
        help="Agent type (random or perfect cheating agent)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations after episodes",
    )
    parser.add_argument(
        "--hrtf-path",
        type=str,
        default="assets/hrtf/test_kemar/",
        help="Path to HRTF dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/demo_recordings",
        help="Output directory for recordings",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Physonic RL Environment Demo")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Agent: {args.agent}")
    print(f"Episodes: {args.episodes}")
    print()

    # Create environment
    if args.env == "localization":
        env = SoundLocalizationEnv(
            audio_duration=1.0,
            use_mel_spectrogram=True,
            hrtf_path=args.hrtf_path,
        )
        print("Task: Predict azimuth angle from binaural audio")
        print(f"Observation shape: {env.observation_space.shape}")
        print(f"Action space: {env.action_space}")

    elif args.env == "classification":
        env = SoundClassificationEnv(
            audio_duration=0.5,
            use_mel_spectrogram=True,
            randomize_position=True,
            hrtf_path=args.hrtf_path,
        )
        print("Task: Classify material from impact sound")
        print(f"Observation shape: {env.observation_space.shape}")
        print(f"Action space: {env.action_space}")
        print(f"Classes: {env.MATERIAL_CLASSES}")

    elif args.env == "selective":
        env = SelectiveResponseEnv(
            target_materials=["glass"],
            audio_duration=0.5,
            target_probability=0.3,
            hrtf_path=args.hrtf_path,
        )
        print("Task: Respond to glass sounds, ignore others")
        print(f"Observation shape: {env.observation_space.shape}")
        print(f"Action space: {env.action_space} (0=IGNORE, 1=RESPOND)")
        print(f"Target materials: {env.target_materials}")
        print(f"Target probability: {env.target_probability:.0%}")

    print()

    # Create recorder
    recorder = EpisodeRecorder(
        output_dir=args.output_dir,
        save_audio=True,
        save_observations=True,
    )

    # Run agent
    print(f"Running {args.agent} agent...")
    print("-" * 60)

    if args.agent == "random":
        rewards = run_random_agent(env, args.episodes, recorder)
    else:  # perfect
        if args.env == "localization":
            rewards = run_perfect_agent_localization(env, args.episodes, recorder)
        elif args.env == "classification":
            rewards = run_perfect_agent_classification(env, args.episodes, recorder)
        elif args.env == "selective":
            rewards = run_perfect_agent_selective(env, args.episodes, recorder)

    print("-" * 60)
    print(f"\nResults:")
    print(f"  Mean reward: {np.mean(rewards):.3f}")
    print(f"  Std reward: {np.std(rewards):.3f}")
    print(f"  Min reward: {np.min(rewards):.3f}")
    print(f"  Max reward: {np.max(rewards):.3f}")

    # Environment-specific metrics
    if hasattr(env, "get_metrics"):
        metrics = env.get_metrics()
        print(f"\nEnvironment Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

    # Visualizations
    if args.visualize:
        print(f"\nGenerating visualizations...")
        import matplotlib.pyplot as plt

        # Plot rewards
        fig_rewards = plot_metrics(rewards, title=f"{args.env.title()} - Episode Rewards")
        fig_rewards.savefig(f"{args.output_dir}/rewards.png", dpi=150)
        print(f"  Saved rewards plot to {args.output_dir}/rewards.png")
        plt.close(fig_rewards)

        # Plot confusion matrix (for classification/selective)
        if hasattr(env, "get_confusion_matrix"):
            cm = env.get_confusion_matrix()
            if args.env == "classification":
                labels = env.MATERIAL_CLASSES
            else:
                labels = ["Distractor", "Target"]

            fig_cm = plot_confusion_matrix(cm, labels, title=f"{args.env.title()} - Confusion Matrix")
            fig_cm.savefig(f"{args.output_dir}/confusion_matrix.png", dpi=150)
            print(f"  Saved confusion matrix to {args.output_dir}/confusion_matrix.png")
            plt.close(fig_cm)

        # Plot first episode summary
        episode_path = f"{args.output_dir}/episode_00000"
        if Path(episode_path).exists():
            fig_summary = plot_episode_summary(episode_path)
            fig_summary.savefig(f"{args.output_dir}/episode_summary.png", dpi=150)
            print(f"  Saved episode summary to {args.output_dir}/episode_summary.png")
            plt.close(fig_summary)

    print(f"\nRecordings saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
