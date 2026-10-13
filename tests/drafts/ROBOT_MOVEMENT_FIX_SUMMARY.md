# Robot Movement Fix Summary

## Problem

User reported: "seems like the robot cannot move in reverse direction, noticed this when it hits a wall"

Initial investigation revealed the robot could barely move at all (only 0.02m in 20 steps), regardless of direction.

## Root Causes

### 1. Missing `mj_forward()` Call (CRITICAL)
**File**: `physonic/envs/data_center_nav.py`
**Line**: 309 (after setting qpos)

After setting `data.qpos[0:2]` in `reset()`, the code didn't call `mujoco.mj_forward()` to update derived quantities like `xpos`. This caused a 18m position mismatch between `qpos` (joint positions) and `xpos` (body positions), leading to immediate out-of-bounds termination.

**Symptom**: All episodes terminated after step 1 with `terminated=True`

### 2. Wrong Robot Start Position (CRITICAL)
**File**: `physonic/environments/dynamic_scene.py`
**Line**: 358-361

The robot body was created with `pos="(random_x, random_y, 0.6)"` from `spec.start_position`. This made the slide joints act as **offsets** rather than **absolute positions**:

```
xpos = start_position + qpos
xpos = [5.5, 17.1, 0.6] + [21.57, 11.34, 0] = [27.07, 28.44, 0.6] (OUT OF BOUNDS!)
```

**Symptom**: Setting `qpos[0:2]` to valid coordinates resulted in out-of-bounds `xpos` values

### 3. Position Control Instead of Velocity Control (MODERATE)
**File**: `physonic/environments/dynamic_scene.py`
**Lines**: 227-229

Motors were using default `<motor>` actuators which interpret control signals as **target positions**, not velocities. With `ctrl[0] = 5.0`, the robot tried to slowly reach position `x = 5.0 meters` via a spring-damper system.

**Symptom**: Robot moved only 0.02m in 20 steps despite velocity commands

## Fixes Applied

### Fix 1: Added `mj_forward()` Call
**File**: `physonic/envs/data_center_nav.py` (line 312)

```python
# Set random robot start position (avoiding obstacles)
self.robot_start_position = self._sample_free_position()
self.data.qpos[0:2] = self.robot_start_position[:2]  # x, y

# Forward kinematics to update derived quantities (xpos, xvel, etc.)
mujoco.mj_forward(self.model, self.data)  # ← ADDED

# Set random goal position
self.goal_position = self._sample_free_position()
```

**Result**: `qpos` and `xpos` now match perfectly (0.000000m mismatch)

### Fix 2: Changed Robot Start Position to Origin
**File**: `physonic/environments/dynamic_scene.py` (lines 353-363)

```python
# Add moving robots
for i in range(num_robots):
    # For RL environments, robot starts at origin (0, 0, 0.6)
    # so qpos[0:2] directly controls absolute position
    # (not offsets from start_position)
    robot = MovingRobotSpec(
        name=f"robot_{i+1}",
        start_position=(0.0, 0.0, 0.6),  # ← CHANGED from random position
        end_position=(0.0, 0.0, 0.6),    # Not used in RL
        speed=0.5,
        mass=120.0,
    )
    moving_robots.append(robot)
```

**Result**: `qpos` now directly controls absolute position

### Fix 3: Changed to Velocity Motors
**File**: `physonic/environments/dynamic_scene.py` (lines 225-229)

**Before**:
```xml
<motor joint="{robot.name}_slide_x" gear="200" ctrllimited="true" ctrlrange="-10 10" />
```

**After**:
```xml
<!-- Velocity motors for direct velocity control -->
<!-- kv parameter sets damping (higher = more responsive, lower = smoother) -->
<velocity joint="{robot.name}_slide_x" kv="100" ctrllimited="true" ctrlrange="-10 10" />
```

**Result**: Control signals now directly command velocities

### Fix 4: Robot Physics Improvements (Bonus)
**File**: `physonic/environments/dynamic_scene.py` (lines 142-144)

Also reduced joint damping (10 → 2) for less resistance:
```xml
<joint name="{spec.name}_slide_x" type="slide" axis="1 0 0" limited="false" damping="2" />
<joint name="{spec.name}_slide_y" type="slide" axis="0 1 0" limited="false" damping="2" />
```

## Test Results

### Before Fixes
```
Forward (+X)      : Moved (+0.02, +0.00)m  ❌
Backward (-X)     : Moved (-0.02, +0.00)m  ❌
Right (+Y)        : Moved (+0.00, +0.02)m  ❌
Left (-Y)         : Moved (+0.00, -0.02)m  ❌
Average displacement: 0.02m per 20 steps
Max speed: 0.01 m/s
```

All episodes terminated after step 1 due to out-of-bounds position.

### After Fixes
```
Forward (+X)      : Moved (+7.74, +0.00)m  ✅
Backward (-X)     : Moved (-8.24, +0.00)m  ✅ (REVERSE WORKS!)
Right (+Y)        : Moved (+0.00, +8.24)m  ✅
Left (-Y)         : Moved (+0.00, -8.24)m  ✅
Forward-Right     : Moved (+5.77, +5.77)m  ✅
Backward-Left     : Moved (-5.77, -5.77)m  ✅ (REVERSE WORKS!)
Average displacement: 8.13m per 20 steps
Max speed: 4.06 m/s (81% of commanded 5 m/s)
```

**Wall Escape Test**: ✅ Successfully escaped 10.07m from wall

## Impact

### Training Performance
- **Before**: Episodes terminated immediately, no training possible
- **After**: Full bidirectional movement, training can proceed normally

### Movement Capability
- **Before**: 0.02m in 20 steps (0.01 m/s)
- **After**: 8.13m in 20 steps (4.06 m/s average, 81% efficiency)
- **Reverse movement**: Fully functional (-8.24m in 20 steps)

### Episode Success Rate
- **Before**: 0% (all episodes terminated at step 1)
- **After**: Normal success rate based on policy performance

## Files Modified

1. `physonic/envs/data_center_nav.py`
   - Added `mujoco.mj_forward()` call after setting qpos (line 312)

2. `physonic/environments/dynamic_scene.py`
   - Changed robot start_position from random to (0, 0, 0.6) (lines 353-363)
   - Changed from `<motor>` to `<velocity>` actuators (lines 225-229)
   - Reduced joint damping from 10 to 2 (lines 142-144)

## Configuration Summary

**Final Robot Configuration**:
- Joint damping: 2.0 (reduced from 10.0) - 5× less resistance
- Velocity motors: kv=100 (x/y), kv=50 (rotation)
- Action scaling: ×5.0 - maps [-1, 1] actions to [-5, 5] m/s
- Control range: [-10, 10] m/s (motor limits)
- Mass: 120.0 kg
- Size: 0.6m sphere (radius 0.3m)

**Performance**:
- Commanded velocity: 5.0 m/s
- Actual average velocity: 4.06 m/s
- Efficiency: 81% (good for damped system)

## Diagnostic Tests Created

Created comprehensive test suite for debugging:

1. `test_motor_connection.py` - Tests if motors respond to ctrl commands
2. `test_env_step_debug.py` - Debug env.step() movement
3. `test_body_names.py` - List all bodies and check robot_body_id
4. `test_robot_movement.py` - Full bidirectional movement test (main test)

## Lessons Learned

### MuJoCo Best Practices

1. **Always call `mj_forward()` after modifying `qpos`**: MuJoCo doesn't automatically update derived quantities
2. **Set body start positions to origin for RL**: Makes qpos directly control absolute position
3. **Use `<velocity>` actuators for velocity control**: Default `<motor>` is position control
4. **Joint positions are cumulative**: `xpos = body.pos + joint_offsets`

### Debugging Strategy

1. **Check position consistency**: Verify `qpos` and `xpos` match expectations
2. **Test zero control first**: Rule out unwanted drift/forces
3. **Test opposite directions**: Verify motors are connected correctly
4. **Check termination conditions**: Early termination can mask movement issues

## Future Considerations

### Potential Improvements
- Fine-tune `kv` damping parameter for optimal responsiveness vs stability
- Consider adding position limits (currently unlimited slide joints)
- Add torque limits for more realistic motor constraints
- Consider adding friction/drag for more realistic deceleration

### Performance Notes
- 81% velocity efficiency is acceptable for damped system
- Joint damping=2 provides good balance (responsive but stable)
- kv=100 provides responsive control without instability
