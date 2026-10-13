# Reward Signal Analysis for Audio-Based Navigation

## Current Reward Structure

```python
# physonic/envs/data_center_nav.py:343-361

# Distance penalty (every step)
reward = -0.01 * distance

# Goal reached bonus
if distance < 1.0:
    reward += 10.0

# Collision penalty
if robot_state[2] < 0.1:
    reward -= 5.0

# Out of bounds penalty
if not self._in_bounds(robot_state[:2]):
    reward -= 10.0
```

---

## Critical Issues

### 🚨 Issue 1: **Goal has NO sound source**

**Problem**: The robot is supposed to navigate using audio, but the goal position doesn't emit any sound!

**Current behavior**:
- Robot hears ~24 server racks (all static ambient noise)
- Goal is a silent 3D coordinate
- No acoustic cue to guide navigation

**Consequence**: The robot **cannot use audio** to find the goal. It can only:
- Use ultrasonic sensors (obstacle avoidance)
- Use proprioceptive state (position, velocity)
- Random exploration until it stumbles upon the goal

**This defeats the purpose of audio-based navigation!**

---

### 🚨 Issue 2: **No reward for using audio**

**Problem**: Reward is purely distance-based (ground truth). No incentive to use audio observations.

**Current reward dependencies**:
- ✅ Distance to goal (ground truth - requires knowing goal position)
- ✅ Collision detection (z-position)
- ✅ Bounds checking (x, y position)
- ❌ Audio observation (not used at all)
- ❌ Ultrasonic sensors (not rewarded explicitly)

**Consequence**: The policy might learn to ignore `audio_codes` entirely and only use `state` + `ultrasonic`.

---

### 🚨 Issue 3: **Weak collision penalty**

**Problem**: Collision penalty is too small relative to distance penalty accumulation.

**Math**:
- Distance penalty: `-0.01 * distance` per step
- At 30m distance: `-0.3` per step
- Over 500 steps: `-150` total
- Single collision: `-5.0`

**Consequence**: Robot might learn aggressive collision-prone shortcuts because:
- Collision penalty `-5.0` << cumulative distance penalty `-150`
- Fast, risky navigation is rewarded over safe, cautious navigation

---

### 🚨 Issue 4: **No progress shaping**

**Problem**: Only absolute distance is rewarded, not progress towards goal.

**Current**:
```python
reward = -0.01 * distance  # Penalize being far away
```

**Better**:
```python
# Reward getting closer
delta_distance = previous_distance - current_distance
reward = delta_distance  # Positive if moving closer, negative if moving away
```

**Consequence**: Robot doesn't get immediate feedback for good actions (moving towards goal).

---

### 🚨 Issue 5: **No exploration incentive**

**Problem**: Pure exploitation (distance minimization) without exploration.

**Consequence**:
- Robot might get stuck in local minima
- Won't discover efficient paths around obstacles
- Early training will be very slow (random until lucky)

---

## Proposed Refactoring

### Option A: **Add Goal Sound Source** (Recommended)

Make the goal emit a distinctive sound that the robot can locate:

```python
class DataCenterNavigationEnv:
    def reset(self, ...):
        # Set random goal position
        self.goal_position = self._sample_free_position()

        # Add goal sound source (alert beep)
        goal_sound = ContinuousSoundSpec(
            source_id="goal_beacon",
            sound_type="beep",  # Distinctive from server hum
            fundamental_freq=1000.0,  # 1kHz beep (very different from 120Hz servers)
            harmonics=[1.0],
            noise_amount=0.0,
            modulation_freq=2.0,  # 2Hz pulsing
            modulation_depth=0.5,
            position=tuple(self.goal_position),
            volume=1.0,
        )

        # Update audio generator with goal sound
        self.audio_generator.add_dynamic_sound(goal_sound)
```

**Benefits**:
- Robot can use audio to locate goal
- Justifies audio-based navigation task
- More realistic (e.g., responding to equipment alarm)

---

### Option B: **Progress-Based Reward Shaping**

Replace absolute distance penalty with progress rewards:

```python
def step(self, action):
    # ... physics step ...

    # Calculate progress
    current_distance = self._distance_to_goal()
    delta_distance = self.previous_distance - current_distance

    # Progress reward (positive if moving closer)
    reward = 10.0 * delta_distance  # Scale to make it significant

    # Goal reached bonus
    if current_distance < 1.0:
        reward += 100.0  # Larger bonus
        terminated = True

    # Collision penalty (stronger)
    if robot_state[2] < 0.1:
        reward -= 20.0  # 4x stronger
        terminated = True

    # Out of bounds penalty (stronger)
    if not self._in_bounds(robot_state[:2]):
        reward -= 50.0  # 5x stronger
        terminated = True

    # Store for next step
    self.previous_distance = current_distance

    return obs, reward, terminated, truncated, info
```

**Benefits**:
- Immediate feedback for good actions
- Scales better across different room sizes
- Stronger penalties for dangerous behavior

---

### Option C: **Audio-Localization Auxiliary Reward**

Add explicit reward for accurate audio-based localization:

```python
def step(self, action):
    # ... standard navigation rewards ...

    # Auxiliary: Reward if robot is heading towards loudest sound
    if self.goal_sound_enabled:
        # Get azimuth to goal
        robot_heading = robot_state[5]
        goal_direction = np.arctan2(
            self.goal_position[1] - robot_state[1],
            self.goal_position[0] - robot_state[0]
        )

        # Heading error
        heading_error = np.abs(np.arctan2(
            np.sin(goal_direction - robot_heading),
            np.cos(goal_direction - robot_heading)
        ))

        # Bonus for facing goal (0 to 1.0)
        heading_bonus = 1.0 * (1.0 - heading_error / np.pi)
        reward += heading_bonus
```

**Benefits**:
- Explicitly rewards using audio for localization
- Helps guide early training
- Can be annealed (reduced) over time as policy improves

---

### Option D: **Sparse Reward Only**

Simplify to pure goal-reaching task:

```python
def step(self, action):
    # ... physics step ...

    # Sparse reward: only at goal
    reward = 0.0

    if distance < 1.0:
        reward = 1.0
        terminated = True

    # Penalties only for failures
    if collision or out_of_bounds:
        reward = -1.0
        terminated = True

    return obs, reward, terminated, truncated, info
```

**Benefits**:
- Forces policy to learn from scratch
- No hand-crafted shaping bias
- More general (doesn't assume distance metric)

**Drawbacks**:
- Much slower training
- Requires more exploration

---

## Recommended Solution

### **Hybrid Approach: Goal Sound + Progress Shaping**

```python
class DataCenterNavigationEnv:
    def __init__(self, ..., use_goal_sound=True, reward_shaping="progress"):
        self.use_goal_sound = use_goal_sound
        self.reward_shaping = reward_shaping
        self.previous_distance = None

    def reset(self, ...):
        # ... existing reset ...

        # Add goal sound source
        if self.use_goal_sound:
            self._add_goal_sound(self.goal_position)

        # Initialize previous distance for progress tracking
        self.previous_distance = self._distance_to_goal()

        return obs, info

    def step(self, action):
        # ... physics step ...

        current_distance = self._distance_to_goal()

        # Reward shaping
        if self.reward_shaping == "progress":
            # Progress-based (recommended)
            delta_distance = self.previous_distance - current_distance
            reward = 10.0 * delta_distance
        elif self.reward_shaping == "distance":
            # Original distance penalty
            reward = -0.01 * current_distance
        else:  # "sparse"
            # No shaping
            reward = 0.0

        # Goal reached (large bonus)
        if current_distance < 1.0:
            reward += 100.0
            terminated = True

        # Collision (strong penalty)
        if robot_state[2] < 0.1:
            reward -= 20.0
            terminated = True

        # Out of bounds (stronger penalty)
        if not self._in_bounds(robot_state[:2]):
            reward -= 50.0
            terminated = True

        # Store for next step
        self.previous_distance = current_distance

        return obs, reward, terminated, truncated, info

    def _add_goal_sound(self, position):
        """Add distinctive sound at goal position."""
        goal_sound = ContinuousSoundSpec(
            source_id="goal_beacon",
            sound_type="beep",
            fundamental_freq=1000.0,  # 1kHz (vs 120Hz servers)
            harmonics=[1.0, 0.5],
            noise_amount=0.0,
            modulation_freq=2.0,  # 2Hz pulsing beep
            modulation_depth=0.8,  # Strong modulation
            position=tuple(position),
            volume=2.0,  # Louder than servers
        )

        # Add to audio generator's sound list
        self.audio_generator.continuous_sounds.append(goal_sound)
```

---

## Ablation Studies Needed

Test different configurations to find best performance:

| Configuration | Goal Sound | Reward Shaping | Expected Performance |
|---------------|-----------|----------------|---------------------|
| **Current** | ❌ No | Distance penalty | Poor (no audio cue) |
| **A** | ✅ Yes | Distance penalty | Medium (audio available but weak shaping) |
| **B** | ❌ No | Progress-based | Medium (better shaping, no audio) |
| **C** | ✅ Yes | Progress-based | **Best** (audio + good shaping) |
| **D** | ✅ Yes | Sparse only | Slow but general |

---

## Training Metrics to Track

Add to tensorboard logging:

```python
# Episode-level metrics
- "episode/success_rate": goal reached / total episodes
- "episode/collision_rate": collisions / total episodes
- "episode/avg_distance_to_goal": mean final distance
- "episode/avg_steps": mean steps per episode

# Step-level metrics
- "step/distance_to_goal": current distance
- "step/progress": delta distance (positive = good)
- "step/collision": binary collision indicator
- "step/audio_rms": RMS of audio observation (is audio being used?)
- "step/ultrasonic_min": minimum ultrasonic distance (obstacle proximity)

# Reward decomposition
- "reward/total": total reward
- "reward/progress": progress component
- "reward/goal_bonus": goal reaching bonus
- "reward/collision_penalty": collision penalty
- "reward/bounds_penalty": out-of-bounds penalty
```

---

## Conclusion

**Critical changes needed**:
1. ✅ **Add goal sound source** - Essential for audio-based navigation
2. ✅ **Use progress-based shaping** - Better than absolute distance
3. ✅ **Increase collision penalties** - Encourage safe navigation
4. ✅ **Make reward shaping configurable** - Enable ablation studies

**Without these changes**, the current environment is:
- Not truly audio-based (goal is silent)
- Poorly shaped (weak progress signal)
- Potentially unsafe (weak collision penalty)

Implementing these changes will make the task:
- Actually solvable with audio
- Faster to train
- More robust to hyperparameter choices
