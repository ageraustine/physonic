# Scripts Directory

This directory contains all executable scripts for the Physonic project, organized by function.

## Directory Structure

```
scripts/
├── baselines/         # Baseline agents for RL environments
├── demos/             # Demo scripts showcasing features
├── examples/          # Simple usage examples
├── training/          # Training scripts for RL agents
├── utils/             # Utility and setup scripts
└── visualization/     # Visualization tools
```

## Baselines

Baseline agent implementations for benchmarking RL environments:

- `baseline_random.py` - Random action selection
- `baseline_heuristic.py` - Heuristic-based decision making
- `baseline_selective_always_respond.py` - Always respond (for selective response task)
- `baseline_selective_never_respond.py` - Never respond (for selective response task)
- `baseline_selective_random.py` - Random response (for selective response task)

**Usage**:
```bash
python3 scripts/baselines/baseline_random.py
```

## Demos

Interactive demonstrations of Physonic capabilities:

- `demo_environment_gen.py` - Procedural environment generation demo
- `demo_rl_env.py` - RL environment testing with various agents
- `generate_data_center_layouts.py` - Data center layout generation examples

**Usage**:
```bash
# Test RL environments
python3 scripts/demos/demo_rl_env.py --env localization --agent perfect --episodes 10

# Generate data center layouts
python3 scripts/demos/generate_data_center_layouts.py
```

## Examples

Simple usage examples for getting started:

- `example.py` - Basic Physonic usage example

**Usage**:
```bash
python3 scripts/examples/example.py
```

## Training

Training scripts for RL agents:

- `train_ppo_data_center.py` - PPO agent for data center navigation
- `train_selective_response.py` - Training for selective response task

**Usage**:
```bash
# Quick test run
python3 scripts/training/train_ppo_data_center.py --total-timesteps 2000 --n-envs 1

# Full training
python3 scripts/training/train_ppo_data_center.py --total-timesteps 1000000 --n-envs 4
```

## Utilities

Utility and setup scripts:

- `setup_hrtf_test.py` - HRTF dataset setup and validation
- `organize_outputs.py` - Output directory organization

**Usage**:
```bash
# HRTF setup
python3 scripts/utils/setup_hrtf_test.py

# Organize outputs (dry run)
python3 scripts/utils/organize_outputs.py

# Organize outputs (execute)
python3 scripts/utils/organize_outputs.py --execute
```

**`organize_outputs.py` details**:
- Finds all `.wav`, `_telemetry.npz`, `_events.json`, `_registry.json` files
- Categorizes them based on filename (mono, spatial, reverb, multi)
- Moves them to appropriate `outputs/` subdirectories
- Cleans up empty old directories

## Visualization

Tools for visualizing environments and results:

- `view_environment.py` - 3D environment viewer with multi-angle rendering

**Usage**:
```bash
# View existing scene
python3 scripts/visualization/view_environment.py --xml outputs/demo_environments/data_center.xml

# Generate and view
python3 scripts/visualization/view_environment.py --scenario maintenance --layout hot_aisle
```

---

## Project Organization Policy

**Important**: All Python files must be organized as follows:

1. **Library code** → `physonic/` directory
   - Core modules, algorithms, utilities
   - Example: `physonic/environments/layout.py`

2. **Executable scripts** → `scripts/` directory (this directory)
   - Organized into subdirectories by function
   - Example: `scripts/training/train_ppo_data_center.py`

3. **Tests** → `tests/` directory
   - Mirror structure of `physonic/` directory
   - Example: `tests/test_environments/test_layout.py`

**No Python files should be at the project root** (except `setup.py` if needed for packaging).

---

## Adding New Scripts

When adding new scripts, follow these guidelines:

1. **Choose the right subdirectory**:
   - `demos/` - Interactive showcases of features
   - `training/` - RL agent training scripts
   - `baselines/` - Reference implementations for benchmarking
   - `visualization/` - Tools for rendering/plotting
   - `utils/` - Setup, organization, maintenance scripts
   - `examples/` - Simple, self-contained usage examples

2. **Add docstrings**:
   - Module-level docstring explaining what the script does
   - Command-line argument help text
   - Usage examples in the docstring

3. **Update this README**:
   - Add entry in the appropriate section
   - Include usage example

4. **Make executable** (optional):
   ```bash
   chmod +x scripts/subdirectory/your_script.py
   ```

5. **Add to main README.md** if it's a major feature

---

## Running Scripts

All scripts should be run from the project root:

```bash
# Good (from project root)
cd /path/to/physonic_mvp
python3 scripts/training/train_ppo_data_center.py

# Bad (from within scripts/)
cd scripts/training
python3 train_ppo_data_center.py  # May have import issues
```

This ensures consistent Python module paths and relative file access.
