"""Visualize the reward signal issue in data center navigation."""

import matplotlib.pyplot as plt
import numpy as np

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Reward Signal Analysis: Critical Issues", fontsize=16, fontweight='bold')

# Issue 1: No goal sound source
ax = axes[0, 0]
room_x, room_y = 30, 25

# Plot server racks (with sound)
rack_positions = [
    (5, 3), (10, 3), (15, 3), (20, 3), (25, 3),
    (5, 8), (10, 8), (15, 8), (20, 8), (25, 8),
    (5, 13), (10, 13), (15, 13), (20, 13), (25, 13),
    (5, 18), (10, 18), (15, 18), (20, 18), (25, 18),
]
for pos in rack_positions:
    ax.scatter(*pos, c='gray', s=200, marker='s', alpha=0.6, edgecolors='black', linewidths=1)
    # Sound waves
    circle = plt.Circle(pos, 2.0, fill=False, color='blue', linestyle='--', alpha=0.3, linewidth=1)
    ax.add_patch(circle)

# Robot
robot_pos = (8, 12)
ax.scatter(*robot_pos, c='orange', s=400, marker='o', edgecolors='black', linewidths=2, zorder=10)
ax.text(robot_pos[0], robot_pos[1]-2, 'Robot', ha='center', fontsize=10, fontweight='bold')

# Goal (SILENT!)
goal_pos = (22, 18)
ax.scatter(*goal_pos, c='red', s=500, marker='*', edgecolors='black', linewidths=2, zorder=10)
ax.text(goal_pos[0], goal_pos[1]+2, 'Goal (SILENT!)', ha='center', fontsize=11,
        fontweight='bold', color='red')

# Arrow showing distance reward
ax.annotate('', xy=goal_pos, xytext=robot_pos,
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(15, 16, 'Distance-based\nreward only', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

ax.set_xlim(0, room_x)
ax.set_ylim(0, room_y)
ax.set_xlabel('X (meters)', fontsize=10)
ax.set_ylabel('Y (meters)', fontsize=10)
ax.set_title('Issue 1: Goal Has NO Sound Source!\n(Robot cannot use audio to navigate)',
             fontsize=11, fontweight='bold', color='red')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Add legend
ax.scatter([], [], c='gray', s=200, marker='s', alpha=0.6, edgecolors='black', linewidths=1, label='Server racks (120Hz hum)')
ax.scatter([], [], c='orange', s=400, marker='o', edgecolors='black', linewidths=2, label='Robot')
ax.scatter([], [], c='red', s=500, marker='*', edgecolors='black', linewidths=2, label='Goal (silent)')
ax.plot([], [], 'b--', alpha=0.3, linewidth=1, label='Sound propagation')
ax.legend(loc='upper left', fontsize=8)

# Issue 2: Weak collision penalty
ax = axes[0, 1]
episodes = np.arange(1, 501)

# Distance penalty accumulation
distance_penalty_safe = -0.01 * 20 * episodes  # Safe path, 20m average distance
distance_penalty_risky = -0.01 * 15 * episodes  # Risky path, 15m average distance but crashes at step 300
distance_penalty_risky[300:] = distance_penalty_risky[299]  # Episode ends

# Add collision penalty
collision_penalty = -5.0
distance_penalty_risky[300:] += collision_penalty

# Goal bonus
distance_penalty_safe[-1] += 10.0  # Reaches goal at end
distance_penalty_risky[300] += 0.0  # Never reaches goal

ax.plot(episodes, distance_penalty_safe, 'g-', linewidth=2, label='Safe navigation (no collision)')
ax.plot(episodes, distance_penalty_risky, 'r-', linewidth=2, label='Risky navigation (collision @ step 300)')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(300, color='red', linestyle='--', alpha=0.5)
ax.text(310, -70, 'Collision\n(-5.0)', fontsize=9, color='red', fontweight='bold')
ax.set_xlabel('Step', fontsize=10)
ax.set_ylabel('Cumulative Reward', fontsize=10)
ax.set_title('Issue 2: Collision Penalty Too Weak\n(Cumulative distance penalty >> single collision)',
             fontsize=11, fontweight='bold', color='red')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Issue 3: No progress shaping
ax = axes[1, 0]
steps = np.arange(100)

# Current: distance penalty (no immediate feedback)
distance = 20 - 0.15 * steps  # Robot moving closer
current_reward = -0.01 * distance
current_reward[distance < 1.0] += 10.0  # Goal bonus at end

# Proposed: progress shaping (immediate feedback)
delta_distance = np.diff(distance, prepend=distance[0])
proposed_reward = 10.0 * (-delta_distance)  # Positive when getting closer
proposed_reward[distance < 1.0] += 100.0  # Larger goal bonus

ax.plot(steps, current_reward, 'r-', linewidth=2, label='Current (distance penalty)', alpha=0.7)
ax.plot(steps, proposed_reward, 'g-', linewidth=2, label='Proposed (progress reward)', alpha=0.7)
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Step', fontsize=10)
ax.set_ylabel('Reward', fontsize=10)
ax.set_title('Issue 3: No Progress Shaping\n(Current: always negative until goal)',
             fontsize=11, fontweight='bold', color='red')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Issue 4: Observation usage
ax = axes[1, 1]
obs_types = ['Audio\nCodes\n(32×7)', 'Ultrasonic\n(4)', 'State\n(6)']
usage_current = [0, 50, 100]  # Audio not needed, ultrasonic useful, state essential
usage_with_goal_sound = [80, 50, 100]  # Audio essential with goal sound

x = np.arange(len(obs_types))
width = 0.35

bars1 = ax.bar(x - width/2, usage_current, width, label='Current (silent goal)',
               color=['red', 'orange', 'green'], alpha=0.7)
bars2 = ax.bar(x + width/2, usage_with_goal_sound, width, label='With goal sound',
               color=['green', 'orange', 'green'], alpha=0.7)

ax.set_ylabel('Estimated Usage (%)', fontsize=10)
ax.set_title('Issue 4: Audio Not Needed\n(Robot can ignore audio observations)',
             fontsize=11, fontweight='bold', color='red')
ax.set_xticks(x)
ax.set_xticklabels(obs_types, fontsize=9)
ax.set_ylim(0, 110)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Add percentage labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{int(height)}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('outputs/reward_issues_visualization.png', dpi=150, bbox_inches='tight')
print("✅ Saved visualization to: outputs/reward_issues_visualization.png")
plt.close()

# Summary statistics
print("\n" + "="*70)
print("REWARD SIGNAL ANALYSIS SUMMARY")
print("="*70)
print()
print("Critical Issues Found:")
print()
print("1. ❌ Goal has NO sound source")
print("   - Robot cannot use audio for navigation")
print("   - Docstring says 'using acoustic cues' but goal is silent!")
print()
print("2. ❌ Collision penalty too weak (-5.0)")
print("   - Distance penalty over 500 steps: ~-150")
print("   - Single collision: -5.0")
print("   - Ratio: 30:1 (encourages risky behavior)")
print()
print("3. ❌ No progress shaping")
print("   - Current: -0.01 * distance (always negative until goal)")
print("   - No immediate reward for moving towards goal")
print("   - Slow learning, poor credit assignment")
print()
print("4. ❌ Audio observations not needed")
print("   - Robot can navigate using only ultrasonic + state")
print("   - Defeats purpose of audio-based navigation")
print()
print("="*70)
print("Recommended Fix: See REWARD_ANALYSIS.md")
print("="*70)
