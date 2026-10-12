# .claude/ - Claude Code Configuration

Minimal, actionable configuration for Physonic development.

## Structure

```
.claude/
├── settings.json              # Validation thresholds (ITD ±0.1ms, ILD ±2dB)
├── rules/
│   └── phase_2a_spec.md      # Function signatures for spatial audio
├── commands/
│   ├── phase2a-setup.md      # /phase2a-setup - Bootstrap dev environment
│   └── validate-spatial.md   # /validate-spatial - Run validation suite
└── skills/
    ├── spatial-audio-engineer/   # HRTF, binaural rendering, ITD/ILD
    ├── rl-engineer/              # Gymnasium, PPO, reward shaping
    └── dataset-engineer/         # LeRobotDataset, Parquet, HF Hub
```

## Quick Start

```bash
# 1. Read the spec
cat .claude/rules/phase_2a_spec.md

# 2. Set up environment
/phase2a-setup

# 3. Implement with exact function signatures from spec

# 4. Validate
/validate-spatial
```

## Skills Auto-Activate

Ask questions naturally - Claude will invoke the right specialist:

- **Spatial audio questions** → `spatial-audio-engineer` skill
- **RL environment questions** → `rl-engineer` skill
- **Dataset export questions** → `dataset-engineer` skill
