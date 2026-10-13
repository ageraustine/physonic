"""Debug env.step() to understand why robot isn't moving."""

import numpy as np
from physonic.envs.data_center_nav import DataCenterNavigationEnv


def test_env_step_movement():
    """Test robot movement using env.step()."""
    print("=" * 70)
    print("Debugging env.step() Movement")
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
    print(f"Initial robot position: ({robot_pos_initial[0]:.2f}, {robot_pos_initial[1]:.2f})")
    print(f"Model timestep: {env.model.opt.timestep}")
    print(f"Control frequency: {env.control_freq} Hz")

    # Calculate steps per control
    steps_per_control = int(1.0 / (env.control_freq * env.model.opt.timestep))
    print(f"Physics steps per control step: {steps_per_control}")
    print(f"Time per env.step(): {steps_per_control * env.model.opt.timestep:.3f}s")
    print()

    # Test 1: Single step with +X action
    print("Test: 1 env.step() with action=[+1.0, 0.0]")
    action = np.array([+1.0, 0.0])
    print(f"Action: {action}")
    print(f"Expected ctrl: [{action[0]*5.0}, {action[1]*5.0}]")

    obs, reward, terminated, truncated, info = env.step(action)

    robot_pos_after = env.data.xpos[env.robot_body_id].copy()
    displacement = robot_pos_after - robot_pos_initial
    print(f"Robot position after 1 step: ({robot_pos_after[0]:.2f}, {robot_pos_after[1]:.2f})")
    print(f"Displacement: ({displacement[0]:+.4f}, {displacement[1]:+.4f})")
    print(f"Distance: {np.linalg.norm(displacement[:2]):.4f}m")
    print()

    # Test 2: Multiple steps
    print("Test: 20 env.step() with action=[+1.0, 0.0]")
    env.reset(seed=42)
    robot_pos_initial = env.data.xpos[env.robot_body_id].copy()

    action = np.array([+1.0, 0.0])
    for i in range(20):
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 5 == 0:
            robot_pos = env.data.xpos[env.robot_body_id].copy()
            disp = robot_pos - robot_pos_initial
            print(f"  Step {i:2d}: Position = ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), "
                  f"Displacement = ({disp[0]:+.4f}, {disp[1]:+.4f})")

    robot_pos_final = env.data.xpos[env.robot_body_id].copy()
    displacement_total = robot_pos_final - robot_pos_initial
    print(f"\nFinal displacement: ({displacement_total[0]:+.4f}, {displacement_total[1]:+.4f})")
    print(f"Total distance: {np.linalg.norm(displacement_total[:2]):.4f}m")
    print(f"Total time: {20 * steps_per_control * env.model.opt.timestep:.2f}s")
    print(f"Average velocity: {np.linalg.norm(displacement_total[:2]) / (20 * steps_per_control * env.model.opt.timestep):.3f} m/s")
    print(f"Expected velocity: 5.0 m/s (action=1.0, scaling=5.0)")
    print()


if __name__ == "__main__":
    test_env_step_movement()
