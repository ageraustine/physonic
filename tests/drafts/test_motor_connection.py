"""Diagnose motor connection and control issues."""

import numpy as np
from physonic.envs.data_center_nav import DataCenterNavigationEnv


def test_motor_diagnostics():
    """Test if motors are connected and responding to commands."""
    print("=" * 70)
    print("Motor Connection Diagnostics")
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

    # Get initial state
    robot_pos_initial = env.data.xpos[env.robot_body_id].copy()
    robot_qpos_initial = env.data.qpos[:3].copy()  # First 3: x, y, theta

    print(f"Robot body ID: {env.robot_body_id}")
    print(f"Initial position (xpos): {robot_pos_initial}")
    print(f"Initial qpos: {robot_qpos_initial}")
    print()

    # Check control indices
    print(f"Number of actuators: {env.model.nu}")
    print(f"Control vector size: {len(env.data.ctrl)}")

    # Print actuator info
    for i in range(env.model.nu):
        actuator_name = env.model.actuator(i).name
        print(f"  Actuator {i}: {actuator_name}")

    print()

    # Test 1: Apply zero control (should not move much)
    print("Test 1: Zero control (10 steps)")
    env.data.ctrl[:] = 0.0
    for _ in range(10):
        for _ in range(int(1.0 / (env.control_freq * env.model.opt.timestep))):
            import mujoco
            mujoco.mj_step(env.model, env.data)

    robot_pos_zero = env.data.xpos[env.robot_body_id].copy()
    displacement_zero = robot_pos_zero - robot_pos_initial
    print(f"Position after zero control: {robot_pos_zero}")
    print(f"Displacement: {displacement_zero}")
    print(f"Distance moved: {np.linalg.norm(displacement_zero[:2]):.4f}m")
    print()

    # Reset
    obs, info = env.reset(seed=42)
    robot_pos_initial = env.data.xpos[env.robot_body_id].copy()

    # Test 2: Apply strong positive X control
    print("Test 2: Strong +X control (ctrl[0] = +10.0, 10 steps)")
    for step in range(10):
        env.data.ctrl[0] = 10.0
        env.data.ctrl[1] = 0.0
        for _ in range(int(1.0 / (env.control_freq * env.model.opt.timestep))):
            import mujoco
            mujoco.mj_step(env.model, env.data)

        if step % 3 == 0:
            pos = env.data.xpos[env.robot_body_id].copy()
            print(f"  Step {step}: Position = ({pos[0]:.2f}, {pos[1]:.2f})")

    robot_pos_plus_x = env.data.xpos[env.robot_body_id].copy()
    displacement_plus_x = robot_pos_plus_x - robot_pos_initial
    print(f"Final position: {robot_pos_plus_x}")
    print(f"Displacement: ({displacement_plus_x[0]:+.2f}, {displacement_plus_x[1]:+.2f})")
    print(f"Distance moved: {np.linalg.norm(displacement_plus_x[:2]):.4f}m")
    print()

    # Reset
    obs, info = env.reset(seed=42)
    robot_pos_initial = env.data.xpos[env.robot_body_id].copy()

    # Test 3: Apply strong negative X control (reverse)
    print("Test 3: Strong -X control (ctrl[0] = -10.0, 10 steps)")
    for step in range(10):
        env.data.ctrl[0] = -10.0
        env.data.ctrl[1] = 0.0
        for _ in range(int(1.0 / (env.control_freq * env.model.opt.timestep))):
            import mujoco
            mujoco.mj_step(env.model, env.data)

        if step % 3 == 0:
            pos = env.data.xpos[env.robot_body_id].copy()
            print(f"  Step {step}: Position = ({pos[0]:.2f}, {pos[1]:.2f})")

    robot_pos_minus_x = env.data.xpos[env.robot_body_id].copy()
    displacement_minus_x = robot_pos_minus_x - robot_pos_initial
    print(f"Final position: {robot_pos_minus_x}")
    print(f"Displacement: ({displacement_minus_x[0]:+.2f}, {displacement_minus_x[1]:+.2f})")
    print(f"Distance moved: {np.linalg.norm(displacement_minus_x[:2]):.4f}m")
    print()

    # Analysis
    print("=" * 70)
    print("Analysis")
    print("=" * 70)

    # Check if +X and -X produce opposite movements
    if displacement_plus_x[0] > 0.1 and displacement_minus_x[0] < -0.1:
        print("✅ X-axis motor working correctly (opposite movements)")
    elif abs(displacement_plus_x[0]) < 0.1 and abs(displacement_minus_x[0]) < 0.1:
        print("❌ X-axis motor NOT working (no movement)")
    else:
        print("⚠️  X-axis motor behavior unexpected")
        print(f"   +X displacement: {displacement_plus_x[0]:+.2f}m")
        print(f"   -X displacement: {displacement_minus_x[0]:+.2f}m")

    # Check for unwanted Y-axis movement
    if abs(displacement_plus_x[1]) < 0.5 and abs(displacement_minus_x[1]) < 0.5:
        print("✅ No unwanted Y-axis drift")
    else:
        print("⚠️  Unexpected Y-axis drift detected")
        print(f"   +X test Y drift: {displacement_plus_x[1]:+.2f}m")
        print(f"   -X test Y drift: {displacement_minus_x[1]:+.2f}m")


if __name__ == "__main__":
    test_motor_diagnostics()
