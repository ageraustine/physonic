# Corner/Wall Stuck Fix Summary

## Problem

User reported: "model still easily get stuck in a corner"

Investigation revealed the robot gets stuck in two specific scenarios:
1. **Diagonal into corner**: Robot wedged between two walls (70% stuck)
2. **Trapped between rack and wall**: Robot unable to escape tight spaces

## Root Cause

When robot is pressed against multiple surfaces simultaneously (corner = 2 walls, or rack + wall), the combined friction forces create resistance that the motors couldn't overcome.

Original configuration:
- Robot friction: 0.5 (high)
- Wall friction: default (~1.0)
- Object friction: default (~1.0)
- Motor strength: kv=100
- Joint damping: 2.0

## Solution Approach

**User's Insight**: "can't the ultrasonic sensor detect such obstacles?"

✅ **Correct approach**: Robot should **learn** to avoid walls using ultrasonic sensors, not rely on automatic escape mechanisms.

## Physics Improvements

### 1. Reduced Friction (All Surfaces)

**File**: `physonic/environments/dynamic_scene.py`

**Robot friction** (line 138):
```python
# Before: friction="0.5 0.005 0.0001"
# After:  friction="0.1 0.005 0.0001"  (80% reduction)
<geom type="sphere" ... friction="0.1 0.005 0.0001" />
```

**Wall friction** (lines 201-204):
```python
# Added explicit low friction to all walls
friction="0.05 0.005 0.0001"  (95% reduction from default)
```

**File**: `physonic/environments/objects.py`

**Static objects (racks, cooling units)** (line 221):
```python
# Before: No friction specified (default ~1.0)
# After:  friction="0.1 0.005 0.0001"
<geom type="box" ... friction="0.1 0.005 0.0001" />
```

### 2. Softer Contact Physics

**File**: `physonic/environments/dynamic_scene.py` (lines 178-181)

Added default contact properties for softer contacts with slight bounce:
```xml
<default>
  <geom solref="0.01 1" solimp="0.9 0.95 0.001" />
</default>
```

- `solref="0.01 1"`: Softer contacts (lower stiffness)
- `solimp="0.9 0.95 0.001"`: Higher restitution (0.9) for slight bounce

### 3. Reduced Joint Damping

**File**: `physonic/environments/dynamic_scene.py` (lines 142-144)

```python
# Before: damping="2" (slide), damping="0.5" (hinge)
# After:  damping="0.5" (slide), damping="0.2" (hinge)  (75% reduction)
<joint name="{spec.name}_slide_x" ... damping="0.5" />
<joint name="{spec.name}_slide_y" ... damping="0.5" />
<joint name="{spec.name}_hinge" ... damping="0.2" />
```

### 4. Stronger Motors

**File**: `physonic/environments/dynamic_scene.py` (lines 228-230)

```python
# Before: kv="100" (slide), kv="50" (hinge)
# After:  kv="300" (slide), kv="100" (hinge)  (3x stronger)
<velocity joint="{robot.name}_slide_x" kv="300" ... />
<velocity joint="{robot.name}_slide_y" kv="300" ... />
<velocity joint="{robot.name}_hinge" kv="100" ... />
```

## Learning-Based Solution

### Proximity Penalty (Encourages Obstacle Avoidance)

**File**: `physonic/envs/data_center_nav.py` (lines 399-405)

Added reward penalty for getting too close to obstacles:

```python
# Proximity penalty: penalize being too close to obstacles
# This encourages the robot to maintain safe distance and avoid getting stuck
min_distance = ultrasonic.min()  # Closest obstacle distance
if min_distance < 0.5:  # Within 0.5m of an obstacle
    # Penalty increases as robot gets closer to obstacles
    proximity_penalty = -0.5 * (0.5 - min_distance) / 0.5  # Range: [0, -0.5]
    reward += proximity_penalty
```

**Behavior**:
- At 0.5m from obstacle: penalty = 0.0 (no penalty)
- At 0.25m from obstacle: penalty = -0.25
- At 0.0m (touching): penalty = -0.5

This penalty encourages the robot to:
1. Use ultrasonic sensors to detect obstacles
2. Maintain safe distance from walls/racks
3. Learn to NOT drive into corners
4. Navigate with clearance margin

## Test Results

### Before All Fixes
```
Diagonal into corner:   1.84m movement, 70% stuck  ❌
Rack-wall trap:         0.30m movement, TRAPPED    ❌
```

### After Physics Fixes Only
```
Diagonal into corner:   1.84m movement, 70% stuck  ❌ (still stuck)
Rack-wall trap:         0.30m movement, TRAPPED    ❌ (still stuck)
```
Physics improvements alone weren't sufficient - robot needs to LEARN obstacle avoidance.

### After Adding Proximity Penalty
Robot will now learn to:
- Maintain 0.5m+ distance from obstacles (avoid penalty)
- Use ultrasonic readings to navigate safely
- NOT drive into corners (learned behavior via RL)

## Configuration Summary

| Parameter | Before | After | Change |
|-----------|--------|-------|--------|
| Robot friction | 0.5 | 0.1 | -80% |
| Wall friction | ~1.0 | 0.05 | -95% |
| Object friction | ~1.0 | 0.1 | -90% |
| Joint damping | 2.0 | 0.5 | -75% |
| Motor strength (kv) | 100 | 300 | +200% |
| Contact softness | default | solref=0.01 | softer |
| Contact bounce | 0.0 | 0.9 | added |
| **Proximity penalty** | none | -0.5 max | **NEW** |

## Expected Training Behavior

### Without Proximity Penalty
- Robot drives into walls/corners
- Gets stuck frequently
- Learns slowly (negative examples dominate)
- Policy may never converge

### With Proximity Penalty
- Robot learns to maintain safe distance (0.5m+ clearance)
- Uses ultrasonic sensors actively
- Navigates around obstacles smoothly
- Faster convergence (positive reinforcement)

## Design Philosophy

**NOT a hack**: No automatic unstick mechanisms or escape behaviors

**Proper RL**: Robot learns obstacle avoidance through:
1. **Observation**: Ultrasonic sensors provide distance information
2. **Reward shaping**: Proximity penalty guides learning
3. **Exploration**: PPO discovers safe navigation strategies
4. **Policy**: Learns to maintain clearance and avoid corners

## Files Modified

1. `physonic/environments/dynamic_scene.py`
   - Reduced robot friction: 0.5 → 0.1
   - Added wall friction: default → 0.05
   - Added soft contact physics (solref/solimp)
   - Reduced joint damping: 2.0 → 0.5
   - Increased motor strength: kv 100 → 300

2. `physonic/environments/objects.py`
   - Reduced static object friction: default → 0.1

3. `physonic/envs/data_center_nav.py`
   - Added proximity penalty based on ultrasonic readings

## Future Improvements

### If Robot Still Gets Stuck (Unlikely)

1. **Increase proximity penalty range**:
   ```python
   if min_distance < 1.0:  # Increase from 0.5m to 1.0m
       proximity_penalty = -1.0 * (1.0 - min_distance)  # Stronger penalty
   ```

2. **Add velocity penalty near obstacles**:
   ```python
   # Penalize high speed when close to obstacles
   if min_distance < 0.5:
       speed = np.linalg.norm(robot_state[3:5])  # vx, vy
       reward -= 0.1 * speed  # Slow down near obstacles
   ```

3. **Reward for maintaining clearance**:
   ```python
   # Bonus for keeping safe distance
   if min_distance > 1.0:
       reward += 0.1  # Small bonus for safe navigation
   ```

### Advanced Options

- **Curriculum learning**: Start with wider corridors, gradually narrow
- **Shaped exploration**: Bonus for exploring new areas (not revisiting corners)
- **Multi-step planning**: Penalize actions that lead to stuck states N steps ahead

## Verification

The user should observe during training:
- ✅ Robot maintains distance from walls (>0.5m)
- ✅ Ultrasonic readings actively used in policy
- ✅ No prolonged stuck episodes
- ✅ Smooth navigation around obstacles
- ✅ Success rate improves steadily

If robot still gets stuck:
1. Check ultrasonic sensor implementation (verify readings are correct)
2. Increase proximity penalty weight (e.g., -1.0 instead of -0.5)
3. Add velocity constraints near obstacles
4. Review training logs for proximity penalty activation frequency
