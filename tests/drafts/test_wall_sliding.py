"""Test wall-sliding and getting stuck scenarios."""

import numpy as np
import mujoco
from physonic.envs.data_center_nav import DataCenterNavigationEnv


def test_wall_sliding():
    """Test robot's ability to slide along walls."""
    print("=" * 70)
    print("Testing Wall-Sliding Behavior")
    print("=" * 70)
    print()

    env = DataCenterNavigationEnv(
        room_size=(30, 25, 4),
        num_racks=24,
        use_goal_sound=True,
        reward_shaping="progress",
        seed=42,
    )

    # Test scenarios where robots commonly get stuck
    scenarios = [
        {
            'name': 'Slide Along Left Wall',
            'start': (0.5, 5.0),
            'action': np.array([0.0, 1.0]),  # Try to move along wall (+Y)
            'expected_y': 10.0,  # Should move at least 5m in Y
        },
        {
            'name': 'Hit Wall Head-On Then Slide',
            'start': (5.0, 0.5),
            'action': np.array([1.0, 0.0]),  # Try to move through wall (will hit and stop)
            'expected_y': 0.5,  # Should NOT move much
        },
        {
            'name': 'Diagonal Into Corner',
            'start': (2.0, 2.0),
            'action': np.array([-1.0, -1.0]),  # Move toward corner
            'expected_dist': 2.0,  # Should hit corner and stop/bounce
        },
        {
            'name': 'Parallel to Wall',
            'start': (15.0, 0.5),
            'action': np.array([1.0, 0.0]),  # Move parallel to bottom wall
            'expected_x': 20.0,  # Should move freely
        },
    ]

    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*70}")
        print(f"Start: ({scenario['start'][0]:.1f}, {scenario['start'][1]:.1f})")
        print(f"Action: ({scenario['action'][0]:+.1f}, {scenario['action'][1]:+.1f})")
        print()

        # Reset and place robot
        env.reset(seed=42)
        env.data.qpos[0] = scenario['start'][0]
        env.data.qpos[1] = scenario['start'][1]
        mujoco.mj_forward(env.model, env.data)

        start_pos = env.data.xpos[env.robot_body_id].copy()

        # Run for 30 steps
        positions = []
        stuck_count = 0
        prev_pos = start_pos.copy()

        for step in range(30):
            obs, reward, terminated, truncated, info = env.step(scenario['action'])
            pos = env.data.xpos[env.robot_body_id].copy()
            positions.append(pos[:2].copy())

            # Check if stuck (moved less than 1cm in this step)
            step_movement = np.linalg.norm(pos[:2] - prev_pos[:2])
            if step_movement < 0.01:
                stuck_count += 1

            if step % 10 == 0:
                total_dist = np.linalg.norm(pos[:2] - start_pos[:2])
                print(f"  Step {step:2d}: ({pos[0]:.2f}, {pos[1]:.2f}), "
                      f"Total: {total_dist:.2f}m, Step: {step_movement:.3f}m")

            prev_pos = pos.copy()

            if terminated or truncated:
                print(f"  Episode ended at step {step}")
                break

        final_pos = env.data.xpos[env.robot_body_id].copy()
        total_movement = np.linalg.norm(final_pos[:2] - start_pos[:2])

        # Analysis
        stuck_ratio = stuck_count / 30
        print(f"\nResults:")
        print(f"  Total movement: {total_movement:.2f}m")
        print(f"  Stuck steps: {stuck_count}/30 ({stuck_ratio*100:.1f}%)")

        if stuck_ratio > 0.5:
            print(f"  ❌ STUCK! Robot barely moved ({stuck_ratio*100:.0f}% of steps stuck)")
        elif stuck_ratio > 0.2:
            print(f"  ⚠️  Sluggish movement (stuck {stuck_ratio*100:.0f}% of time)")
        else:
            print(f"  ✅ Good movement (only stuck {stuck_ratio*100:.0f}% of time)")


def test_rack_wall_trap():
    """Test robot trapped between wall and rack."""
    print("\n\n" + "=" * 70)
    print("Testing Rack-Wall Trap Scenario")
    print("=" * 70)
    print()

    env = DataCenterNavigationEnv(
        room_size=(30, 25, 4),
        num_racks=24,
        use_goal_sound=True,
        reward_shaping="progress",
        seed=42,
    )

    env.reset(seed=42)

    # Find a rack position near a wall
    print("Looking for racks near walls...")
    rack_ids = []
    for i in range(env.model.nbody):
        body_name = env.model.body(i).name
        if 'rack' in body_name:
            rack_pos = env.data.xpos[i]
            # Check if near left wall (x < 3) or bottom wall (y < 3)
            if rack_pos[0] < 5 or rack_pos[1] < 5:
                print(f"  Found {body_name} at ({rack_pos[0]:.1f}, {rack_pos[1]:.1f})")
                rack_ids.append((i, body_name, rack_pos.copy()))

    if rack_ids:
        # Pick first rack near wall
        rack_id, rack_name, rack_pos = rack_ids[0]

        # Place robot between rack and wall
        # If rack is near left wall (x < 5), place robot to the left
        if rack_pos[0] < 5:
            trap_pos = (max(0.8, rack_pos[0] - 0.8), rack_pos[1])
            escape_action = np.array([1.0, 0.0])  # Try to move right (away from wall)
            print(f"\nTrapped between left wall and {rack_name}")
        # If rack is near bottom wall (y < 5)
        elif rack_pos[1] < 5:
            trap_pos = (rack_pos[0], max(0.8, rack_pos[1] - 0.8))
            escape_action = np.array([0.0, 1.0])  # Try to move up (away from wall)
            print(f"\nTrapped between bottom wall and {rack_name}")
        else:
            trap_pos = (rack_pos[0] - 0.8, rack_pos[1])
            escape_action = np.array([1.0, 0.0])

        print(f"Robot position: ({trap_pos[0]:.1f}, {trap_pos[1]:.1f})")
        print(f"Rack position: ({rack_pos[0]:.1f}, {rack_pos[1]:.1f})")
        print(f"Trying to escape with action ({escape_action[0]:+.1f}, {escape_action[1]:+.1f})")
        print()

        env.data.qpos[0] = trap_pos[0]
        env.data.qpos[1] = trap_pos[1]
        mujoco.mj_forward(env.model, env.data)

        start_pos = env.data.xpos[env.robot_body_id].copy()

        # Try to escape
        for step in range(50):
            obs, reward, terminated, truncated, info = env.step(escape_action)
            pos = env.data.xpos[env.robot_body_id].copy()

            if step % 10 == 0:
                dist = np.linalg.norm(pos[:2] - start_pos[:2])
                print(f"  Step {step:2d}: ({pos[0]:.2f}, {pos[1]:.2f}), Distance: {dist:.2f}m")

            if terminated or truncated:
                break

        final_pos = env.data.xpos[env.robot_body_id].copy()
        escape_dist = np.linalg.norm(final_pos[:2] - start_pos[:2])

        if escape_dist > 3.0:
            print(f"\n✅ Escaped from trap ({escape_dist:.2f}m)")
        elif escape_dist > 1.0:
            print(f"\n⚠️  Partial escape ({escape_dist:.2f}m, slow)")
        else:
            print(f"\n❌ TRAPPED! Only moved {escape_dist:.2f}m")


if __name__ == "__main__":
    test_wall_sliding()
    test_rack_wall_trap()
