"""Test realistic stuck scenarios during navigation."""

import numpy as np
import mujoco
from physonic.envs.data_center_nav import DataCenterNavigationEnv


def test_random_navigation():
    """Test if robot gets stuck during random navigation."""
    print("=" * 70)
    print("Realistic Navigation Test (Random Actions)")
    print("=" * 70)
    print()

    env = DataCenterNavigationEnv(
        room_size=(30, 25, 4),
        num_racks=24,
        use_goal_sound=True,
        reward_shaping="progress",
        seed=42,
    )

    num_episodes = 5
    total_stuck_count = 0
    total_steps = 0

    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)
        print(f"\nEpisode {episode + 1}:")
        print(f"  Start: ({env.data.xpos[env.robot_body_id][0]:.1f}, {env.data.xpos[env.robot_body_id][1]:.1f})")
        print(f"  Goal:  ({env.goal_position[0]:.1f}, {env.goal_position[1]:.1f})")

        stuck_steps = 0
        prev_pos = env.data.xpos[env.robot_body_id][:2].copy()

        for step in range(200):  # 20 seconds max
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Check if stuck (moved less than 2cm)
            current_pos = env.data.xpos[env.robot_body_id][:2].copy()
            movement = np.linalg.norm(current_pos - prev_pos)

            if movement < 0.02:
                stuck_steps += 1

            prev_pos = current_pos.copy()

            if terminated or truncated:
                print(f"  Ended at step {step + 1} ({'goal reached' if info.get('distance_to_goal', 999) < 1.0 else 'collision/bounds'})")
                break

        stuck_ratio = stuck_steps / (step + 1)
        total_stuck_count += stuck_steps
        total_steps += (step + 1)

        print(f"  Stuck steps: {stuck_steps}/{step + 1} ({stuck_ratio*100:.1f}%)")

    overall_stuck = total_stuck_count / total_steps
    print()
    print("=" * 70)
    print(f"Overall stuck ratio: {total_stuck_count}/{total_steps} ({overall_stuck*100:.1f}%)")

    if overall_stuck < 0.05:
        print("✅ Excellent - robot rarely gets stuck (<5%)")
    elif overall_stuck < 0.15:
        print("⚠️  Acceptable - robot occasionally stuck (5-15%)")
    else:
        print("❌ Problem - robot frequently stuck (>15%)")


def test_goal_navigation():
    """Test navigation toward goal (more realistic)."""
    print("\n\n" + "=" * 70)
    print("Goal-Directed Navigation Test")
    print("=" * 70)
    print()

    env = DataCenterNavigationEnv(
        room_size=(30, 25, 4),
        num_racks=24,
        use_goal_sound=True,
        reward_shaping="progress",
        seed=42,
    )

    obs, info = env.reset(seed=42)

    robot_pos = env.data.xpos[env.robot_body_id][:2]
    goal_pos = env.goal_position[:2]

    print(f"Start: ({robot_pos[0]:.1f}, {robot_pos[1]:.1f})")
    print(f"Goal:  ({goal_pos[0]:.1f}, {goal_pos[1]:.1f})")
    print(f"Distance: {np.linalg.norm(goal_pos - robot_pos):.1f}m")
    print()
    print("Using simple goal-seeking policy (move toward goal)...")
    print()

    stuck_steps = 0
    prev_pos = robot_pos.copy()

    for step in range(200):
        # Simple policy: move toward goal
        robot_pos = env.data.xpos[env.robot_body_id][:2]
        direction = goal_pos - robot_pos
        distance = np.linalg.norm(direction)

        if distance > 0.1:
            direction = direction / distance  # Normalize
        else:
            direction = np.array([0.0, 0.0])

        action = direction
        obs, reward, terminated, truncated, info = env.step(action)

        # Check if stuck
        current_pos = env.data.xpos[env.robot_body_id][:2]
        movement = np.linalg.norm(current_pos - prev_pos)

        if movement < 0.02:
            stuck_steps += 1

        prev_pos = current_pos.copy()

        if step % 20 == 0:
            dist = np.linalg.norm(current_pos - goal_pos)
            print(f"  Step {step:3d}: Position ({current_pos[0]:.1f}, {current_pos[1]:.1f}), Distance: {dist:.1f}m, Stuck: {stuck_steps}")

        if terminated or truncated:
            final_dist = np.linalg.norm(current_pos - goal_pos)
            if final_dist < 1.0:
                print(f"\n✅ Goal reached at step {step + 1}!")
            else:
                print(f"\n❌ Episode ended at step {step + 1} (distance: {final_dist:.1f}m)")
            break

    stuck_ratio = stuck_steps / (step + 1)
    print(f"\nStuck steps: {stuck_steps}/{step + 1} ({stuck_ratio*100:.1f}%)")

    if stuck_ratio < 0.10:
        print("✅ Good navigation - minimal sticking")
    elif stuck_ratio < 0.25:
        print("⚠️  Some issues - moderate sticking")
    else:
        print("❌ Navigation problems - excessive sticking")


if __name__ == "__main__":
    test_random_navigation()
    test_goal_navigation()
