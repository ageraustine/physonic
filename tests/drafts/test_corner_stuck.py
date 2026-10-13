"""Test if robot can escape from corners."""

import numpy as np
import mujoco
from physonic.envs.data_center_nav import DataCenterNavigationEnv


def test_corner_escape():
    """Test robot's ability to escape from corners."""
    print("=" * 70)
    print("Testing Corner Escape Behavior")
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

    # Test 4 corners
    corners = [
        ("Bottom-Left (0,0)", (0.5, 0.5), np.array([1.0, 1.0])),      # Escape diagonal
        ("Bottom-Right (30,0)", (29.5, 0.5), np.array([-1.0, 1.0])),  # Escape diagonal
        ("Top-Left (0,25)", (0.5, 24.5), np.array([1.0, -1.0])),      # Escape diagonal
        ("Top-Right (30,25)", (29.5, 24.5), np.array([-1.0, -1.0])),  # Escape diagonal
    ]

    results = []

    for corner_name, corner_pos, escape_action in corners:
        print(f"\nTest: {corner_name}")
        print(f"Starting position: ({corner_pos[0]:.1f}, {corner_pos[1]:.1f})")
        print(f"Escape action: ({escape_action[0]:+.1f}, {escape_action[1]:+.1f})")

        # Reset and place robot in corner
        env.reset(seed=42)
        env.data.qpos[0] = corner_pos[0]
        env.data.qpos[1] = corner_pos[1]
        mujoco.mj_forward(env.model, env.data)

        start_pos = env.data.xpos[env.robot_body_id].copy()

        # Try to escape for 50 steps (5 seconds)
        for step in range(50):
            obs, reward, terminated, truncated, info = env.step(escape_action)

            if step % 10 == 0:
                pos = env.data.xpos[env.robot_body_id].copy()
                dist = np.linalg.norm(pos[:2] - start_pos[:2])
                print(f"  Step {step:2d}: Position = ({pos[0]:.2f}, {pos[1]:.2f}), "
                      f"Distance from start: {dist:.2f}m")

            if terminated or truncated:
                break

        final_pos = env.data.xpos[env.robot_body_id].copy()
        escape_distance = np.linalg.norm(final_pos[:2] - start_pos[:2])

        results.append({
            'corner': corner_name,
            'escape_distance': escape_distance,
            'success': escape_distance > 5.0  # At least 5m escape
        })

        if escape_distance > 5.0:
            print(f"✅ Escaped {escape_distance:.2f}m")
        elif escape_distance > 2.0:
            print(f"⚠️  Partial escape {escape_distance:.2f}m (slow)")
        else:
            print(f"❌ STUCK! Only moved {escape_distance:.2f}m")

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    successes = sum(1 for r in results if r['success'])
    print(f"Corners escaped: {successes}/4")

    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"{status} {r['corner']:20s}: {r['escape_distance']:.2f}m")

    print()

    if successes == 4:
        print("✅ All corners: Robot can escape from all corners!")
    elif successes >= 2:
        print("⚠️  Some corners: Robot struggles in some corners")
    else:
        print("❌ Robot gets stuck in corners!")

    # Print current physics parameters
    print()
    print("Current Configuration:")
    print(f"  Robot friction: {env.model.geom('robot_1').friction}")
    print(f"  Joint damping: {env.model.joint('robot_1_slide_x').damping[0]}")
    print(f"  Motor kv: (check actuator settings)")


if __name__ == "__main__":
    test_corner_escape()
