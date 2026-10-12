# Utility Scripts

Helper scripts for Physonic development and data management.

## Available Scripts

### `organize_outputs.py`
Organizes scattered output files into the standardized `outputs/` directory structure.

**Usage:**
```bash
# Preview what will be moved (dry run)
python3 scripts/organize_outputs.py

# Actually move files
python3 scripts/organize_outputs.py --execute
```

**What it does:**
- Finds all `.wav`, `_telemetry.npz`, `_events.json`, `_registry.json` files
- Categorizes them based on filename (mono, spatial, reverb, multi)
- Moves them to appropriate `outputs/` subdirectories
- Cleans up empty old directories

**Example output:**
```
Found 24 output files

mono/ (8 files)
  - drop_steel_on_glass.wav
  - drop_steel_on_steel.wav
  - slide_wood_on_rubber.wav
  ... and 5 more

spatial/ (8 files)
  - drop_steel_on_glass_binaural.wav
  ... and 7 more

✓ Moved 24 files
✓ Removed empty directory: outputs_test
```

## Adding New Scripts

When adding utility scripts:
1. Make them executable: `chmod +x scripts/your_script.py`
2. Add shebang: `#!/usr/bin/env python3`
3. Include docstring with usage example
4. Update this README
5. Consider adding to `.gitignore` if they generate temporary files
