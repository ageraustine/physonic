"""Verify motor power and torque improvements."""

import numpy as np
import mujoco
from physonic.envs.data_center_nav import DataCenterNavigationEnv


def test_motor_power():
    """Test motor power against obstacles."""
    print("=" * 70)
    print("Motor Power Verification Test")
    print("=" * 70)
    print()

    env = DataCenterNavigationEnv(
        room_size=(30, 25, 4),
        num_racks=24,
        use_goal_sound=True,
        seed=42,
    )

    env.reset(seed=42)

    # Print motor specifications
    print("Motor Configuration:")
    for i in range(env.model.nu):
        actuator_name = env.model.actuator(i).name
        # Get actuator properties
        print(f"  Actuator {i}: {actuator_name}")
        # Note: gear and kv are in the XML, not easily accessible via Python API

    print()
    print("=" * 70)
    print("Test 1: Push Against Wall (Maximum Force Test)")
    print("=" * 70)

    # Place robot right at wall
    env.data.qpos[0] = 0.6  # Very close to left wall (x=0)
    env.data.qpos[1] = 12.5
    mujoco.mj_forward(env.model, env.data)

    start_pos = env.data.xpos[env.robot_body_id].copy()
    print(f"Start position: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
    print(f"Pushing LEFT with maximum force...")
    print()

    # Try to push through wall with max force
    for step in range(20):
        action = np.array([-1.0, 0.0])  # Full power left (toward wall)
        obs, reward, terminated, truncated, info = env.step(action)

        pos = env.data.xpos[env.robot_body_id].copy()
        if step % 5 == 0:
            print(f"  Step {step:2d}: Position ({pos[0]:.2f}, {pos[1]:.2f})")

    final_pos = env.data.xpos[env.robot_body_id].copy()
    movement = final_pos[0] - start_pos[0]
    print(f"\nResult: Moved {movement:.3f}m toward wall")

    if abs(movement) < 0.1:
        print("✅ Wall holds (robot can't push through)")
    else:
        print(f"⚠️  Robot moved {movement:.3f}m")

    print()
    print("=" * 70)
    print("Test 2: Slide Along Wall (Lateral Movement)")
    print("=" * 70)

    # Place robot at wall, try to slide along it
    env.reset(seed=42)
    env.data.qpos[0] = 0.6  # Close to left wall
    env.data.qpos[1] = 5.0
    mujoco.mj_forward(env.model, env.data)

    start_pos = env.data.xpos[env.robot_body_id].copy()
    print(f"Start position: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
    print(f"Sliding FORWARD along wall...")
    print()

    for step in range(20):
        action = np.array([0.0, 1.0])  # Move forward (along wall)
        obs, reward, terminated, truncated, info = env.step(action)

        pos = env.data.xpos[env.robot_body_id].copy()
        if step % 5 == 0:
            print(f"  Step {step:2d}: Position ({pos[0]:.2f}, {pos[1]:.2f})")

    final_pos = env.data.xpos[env.robot_body_id].copy()
    movement = np.linalg.norm(final_pos[:2] - start_pos[:2])
    print(f"\nResult: Moved {movement:.2f}m along wall")

    if movement > 3.0:
        print("✅ Robot can slide along walls freely")
    else:
        print(f"❌ Robot struggled to slide (only {movement:.2f}m)")

    print()
    print("=" * 70)
    print("Test 3: Escape From Corner (45° Escape)")
    print("=" * 70)

    # Place robot in corner, try to escape at 45°
    env.reset(seed=42)
    env.data.qpos[0] = 0.6
    env.data.qpos[1] = 0.6
    mujoco.mj_forward(env.model, env.data)

    start_pos = env.data.xpos[env.robot_body_id].copy()
    print(f"Start position: ({start_pos[0]:.2f}, {start_pos[1]:.2f}) - CORNER")
    print(f"Escaping at 45° angle (+X, +Y)...")
    print()

    for step in range(30):
        action = np.array([1.0, 1.0])  # Escape diagonal
        obs, reward, terminated, truncated, info = env.step(action)

        pos = env.data.xpos[env.robot_body_id].copy()
        if step % 10 == 0:
            dist = np.linalg.norm(pos[:2] - start_pos[:2])
            print(f"  Step {step:2d}: Position ({pos[0]:.2f}, {pos[1]:.2f}), Distance: {dist:.2f}m")

    final_pos = env.data.xpos[env.robot_body_id].copy()
    escape_dist = np.linalg.norm(final_pos[:2] - start_pos[:2])
    print(f"\nResult: Escaped {escape_dist:.2f}m from corner")

    if escape_dist > 5.0:
        print("✅ Robot can escape from corners!")
    elif escape_dist > 2.0:
        print(f"⚠️  Partial escape ({escape_dist:.2f}m, slow)")
    else:
        print(f"❌ STUCK in corner (only {escape_dist:.2f}m)")

    print()
    print("=" * 70)
    print("Test 4: Free Space Movement (Baseline)")
    print("=" * 70)

    env.reset(seed=42)
    env.data.qpos[0] = 15.0  # Middle of room
    env.data.qpos[1] = 12.5
    mujoco.mj_forward(env.model, env.data)

    start_pos = env.data.xpos[env.robot_body_id].copy()
    print(f"Start position: ({start_pos[0]:.2f}, {start_pos[1]:.2f}) - OPEN SPACE")
    print(f"Moving forward at full speed...")
    print()

    for step in range(20):
        action = np.array([1.0, 0.0])  # Full speed forward
        obs, reward, terminated, truncated, info = env.step(action)

        pos = env.data.xpos[env.robot_body_id].copy()
        if step % 5 == 0:
            dist = pos[0] - start_pos[0]
            print(f"  Step {step:2d}: Position ({pos[0]:.2f}, {pos[1]:.2f}), Moved: {dist:.2f}m")

    final_pos = env.data.xpos[env.robot_body_id].copy()
    free_movement = final_pos[0] - start_pos[0]
    print(f"\nResult: Moved {free_movement:.2f}m in open space")
    print(f"Average speed: {free_movement / 2.0:.2f} m/s")

    if free_movement > 8.0:
        print("✅ Strong motor power in free space")
    else:
        print(f"⚠️  Slow movement ({free_movement:.2f}m in 2 seconds)")

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  - Motor gear: 1000 (slide), 500 (rotation)")
    print("  - Motor kv: 300 (slide), 100 (rotation)")
    print("  - Control range: [-20, 20]")
    print("  - Action scaling: ×10.0")
    print("  - Robot friction: 0.1")
    print("  - Wall friction: 0.05")
    print()


if __name__ == "__main__":
    test_motor_power()
