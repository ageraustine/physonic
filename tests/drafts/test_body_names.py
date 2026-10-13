"""Check which body is robot_body_id."""

import mujoco
from physonic.envs.data_center_nav import DataCenterNavigationEnv


def list_bodies():
    """List all bodies in the model."""
    env = DataCenterNavigationEnv(
        room_size=(30, 25, 4),
        num_racks=24,
        seed=42,
    )

    print("=" * 70)
    print(f"Total bodies: {env.model.nbody}")
    print(f"robot_body_id: {env.robot_body_id}")
    print("=" * 70)
    print()

    # Find robot_1
    robot_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "robot_1")
    print(f"Body 'robot_1' ID: {robot_id}")

    # List first 40 bodies
    print("\nFirst 40 bodies:")
    for i in range(min(40, env.model.nbody)):
        body_name = env.model.body(i).name
        pos = env.data.xpos[i]
        print(f"  Body {i:3d}: '{body_name:20s}' position = ({pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f})")

    print()

    # Check joints
    print(f"Total joints: {env.model.njnt}")
    print("\nFirst 10 joints:")
    for i in range(min(10, env.model.njnt)):
        joint_name = env.model.joint(i).name
        qpos_addr = env.model.jnt_qposadr[i]
        qpos_val = env.data.qpos[qpos_addr]
        print(f"  Joint {i}: '{joint_name:20s}' qpos[{qpos_addr}] = {qpos_val:.2f}")


def test_reset_positions():
    """Check positions after reset."""
    env = DataCenterNavigationEnv(
        room_size=(30, 25, 4),
        num_racks=24,
        seed=42,
    )

    print("\n" + "=" * 70)
    print("Testing reset() positions")
    print("=" * 70)

    # Reset with fixed seed
    obs, info = env.reset(seed=42)

    print(f"\nAfter reset(seed=42):")
    print(f"robot_body_id: {env.robot_body_id}")
    print(f"qpos[0:3]: {env.data.qpos[0:3]}")
    print(f"xpos[{env.robot_body_id}]: {env.data.xpos[env.robot_body_id]}")
    print(f"Robot start position (set in reset): {env.robot_start_position}")
    print(f"Goal position: {env.goal_position}")

    # Check if qpos matches xpos
    qpos_xy = env.data.qpos[0:2]
    xpos_xy = env.data.xpos[env.robot_body_id][0:2]
    mismatch = np.linalg.norm(qpos_xy - xpos_xy)
    print(f"\nPosition mismatch: {mismatch:.6f}m")

    if mismatch > 0.01:
        print("❌ qpos and xpos don't match! This is the problem!")
    else:
        print("✅ qpos and xpos match correctly")


if __name__ == "__main__":
    import numpy as np
    list_bodies()
    test_reset_positions()
