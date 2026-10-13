"""Test robot movement capabilities including reverse direction."""

import numpy as np
from physonic.envs.data_center_nav import DataCenterNavigationEnv


def test_bidirectional_movement():
    """Test that robot can move in all directions including reverse."""
    print("=" * 70)
    print("Testing Robot Bidirectional Movement")
    print("=" * 70)
    print()

    env = DataCenterNavigationEnv(
        room_size=(30, 25, 4),
        num_racks=24,
        use_goal_sound=True,
        reward_shaping="progress",
        seed=42,
    )

    obs, info = env.reset()
    initial_pos = env.data.xpos[env.robot_body_id].copy()
    print(f"Initial robot position: ({initial_pos[0]:.2f}, {initial_pos[1]:.2f})")
    print()

    # Test movement in all cardinal directions
    directions = {
        "Forward (+X)": np.array([1.0, 0.0]),
        "Backward (-X)": np.array([-1.0, 0.0]),
        "Right (+Y)": np.array([0.0, 1.0]),
        "Left (-Y)": np.array([0.0, -1.0]),
        "Forward-Right": np.array([0.7, 0.7]),
        "Backward-Left": np.array([-0.7, -0.7]),
    }

    results = []

    for direction_name, action in directions.items():
        # Reset environment
        obs, info = env.reset(seed=42)
        start_pos = env.data.xpos[env.robot_body_id].copy()

        # Apply action for 20 steps
        steps_taken = 0
        for i in range(20):
            obs, reward, terminated, truncated, info = env.step(action)
            steps_taken += 1
            if terminated or truncated:
                if steps_taken < 20:
                    print(f"  {direction_name}: Early termination at step {steps_taken} "
                          f"(terminated={terminated}, truncated={truncated})")
                break

        end_pos = env.data.xpos[env.robot_body_id].copy()
        displacement = end_pos - start_pos

        results.append({
            'direction': direction_name,
            'action': action,
            'displacement': displacement[:2],
            'distance': np.linalg.norm(displacement[:2])
        })

        print(f"{direction_name:18s}: "
              f"Action=({action[0]:+.1f}, {action[1]:+.1f}) → "
              f"Moved ({displacement[0]:+.2f}, {displacement[1]:+.2f})m, "
              f"Distance: {np.linalg.norm(displacement[:2]):.2f}m")

    print()
    print("=" * 70)
    print("Movement Analysis")
    print("=" * 70)

    # Check if all movements resulted in displacement
    all_moved = all(r['distance'] > 0.1 for r in results)
    if all_moved:
        print("✅ Robot can move in ALL directions including reverse!")
    else:
        print("❌ Some directions failed to produce movement")

    # Check if reverse movement works
    reverse_movements = [r for r in results if '-' in r['direction']]
    reverse_ok = all(r['distance'] > 0.1 for r in reverse_movements)
    if reverse_ok:
        print("✅ Reverse movement works correctly (backward, left, backward-left)")
    else:
        print("❌ Reverse movement not working properly")

    # Check movement consistency
    avg_distance = np.mean([r['distance'] for r in results])
    print(f"\nAverage displacement: {avg_distance:.2f}m per 20 steps")
    print(f"Max speed: {avg_distance / 2.0:.2f} m/s (approx)")

    print()
    print("Configuration:")
    print(f"  - Joint damping: 2.0 (reduced from 10.0)")
    print(f"  - Motor gear: 200 (increased from 100)")
    print(f"  - Action scaling: ×5.0 (increased from ×2.0)")
    print(f"  - Control range: [-5, 5] m/s (increased from [-2, 2])")
    print()


def test_wall_escape():
    """Test that robot can escape when hitting a wall."""
    print("=" * 70)
    print("Testing Wall Escape Behavior")
    print("=" * 70)
    print()

    env = DataCenterNavigationEnv(
        room_size=(30, 25, 4),
        num_racks=24,
        use_goal_sound=True,
        seed=42,
    )

    obs, info = env.reset()

    # Move robot near a wall (close to x=0 boundary)
    env.data.qpos[0] = 1.0  # Near left wall
    env.data.qpos[1] = 12.5  # Center Y
    import mujoco
    mujoco.mj_forward(env.model, env.data)  # Update xpos

    print("Placed robot near left wall at (1.0, 12.5)")
    print("Attempting to escape by moving forward (+X, away from wall)...")
    print()

    # Try to move forward (away from left wall at x=0)
    escape_action = np.array([+1.0, 0.0])  # Forward in +X direction

    positions = []
    for step in range(30):
        obs, reward, terminated, truncated, info = env.step(escape_action)
        pos = env.data.xpos[env.robot_body_id].copy()
        positions.append(pos[:2])

        if step % 10 == 0:
            print(f"Step {step:2d}: Position ({pos[0]:.2f}, {pos[1]:.2f})")

        if terminated or truncated:
            break

    print()

    # Check if robot moved away from wall
    start_x = 1.0
    final_x = positions[-1][0]
    moved_distance = final_x - start_x

    if moved_distance > 1.0:
        print(f"✅ Robot successfully escaped wall! Moved {moved_distance:.2f}m away")
    elif moved_distance > 0.1:
        print(f"⚠️  Robot moved slightly ({moved_distance:.2f}m) but slowly")
    else:
        print(f"❌ Robot could not escape wall (moved only {moved_distance:.2f}m)")

    print()


if __name__ == "__main__":
    test_bidirectional_movement()
    print()
    test_wall_escape()
