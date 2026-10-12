# Contributing to Physonic

Thank you for your interest in contributing to Physonic! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/ageraustine/physonic.git
   cd physonic
   ```
3. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. **Install dependencies** (including dev tools):
   ```bash
   pip install -r requirements.txt
   ```

## Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Your Changes

Follow these guidelines:
- **Code style**: Use `black` for formatting, `isort` for imports
- **Type hints**: Add type annotations for new functions
- **Docstrings**: Use Google-style docstrings for all public functions
- **Tests**: Add tests for new features

### 3. Run Tests
```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=physonic --cov-report=term-missing

# Run specific test module
pytest tests/test_contact/test_rig.py -v
```

### 4. Format Code
```bash
# Auto-format with black
black physonic/ tests/

# Sort imports
isort physonic/ tests/

# Type check (optional but recommended)
mypy physonic/
```

### 5. Commit Your Changes
```bash
git add .
git commit -m "Add feature: brief description"
```

**Commit message guidelines**:
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and PRs liberally after the first line

### 6. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then open a pull request on GitHub with:
- Clear description of changes
- Link to related issues
- Screenshots/audio samples for new features

## Areas for Contribution

### 🎵 Acoustic Materials
Add new materials to `physonic/contact/materials.py`:
- Ceramics (porcelain, tile)
- Fabrics (cloth, carpet)
- Polymers (plastic varieties)
- Natural materials (stone, bamboo)

**Requirements**:
- Physical properties: density, Young's modulus, Poisson's ratio
- Acoustic properties: fundamental frequency, mode ratios, decay time
- Validation: compare against real recordings (if available)

### 🎬 Multi-Object Scenarios
Add scenarios to `physonic/contact/scenarios.py`:
- More complex interactions (pendulum, Newton's cradle)
- Realistic warehouse tasks (forklift loading, pallet stacking)
- Everyday sounds (drawer opening, door closing)

**Requirements**:
- Clear description and use case
- Parameter documentation
- Example usage in docstring

### 🧪 Validation & Testing
Improve test coverage:
- Perceptual validation (synthetic vs. real recordings)
- Physics accuracy (compare telemetry to analytical solutions)
- Edge cases (extreme parameters, boundary conditions)

### 📊 Dataset Tools
Enhance dataset export functionality:
- Add metadata tags (scenario type, complexity, SNR)
- Implement data augmentation (pitch shift, time stretch, noise)
- Create dataset inspection/visualization tools

### 🤖 RL Environments
Implement Gymnasium environments:
- Sound source localization task
- Material classification task
- Contact prediction task

**Requirements**:
- Follow Gymnasium API conventions
- Include reward shaping
- Document observation/action spaces

## Code Style Guidelines

### Python Code
```python
def compute_modal_synthesis(
    events: List[Dict],
    material: Material,
    sample_rate: int = 44100
) -> np.ndarray:
    """Synthesize impact sounds using modal synthesis.

    Args:
        events: List of impact events with 't' (time) and 'J' (impulse)
        material: Material properties defining acoustic response
        sample_rate: Audio sample rate in Hz

    Returns:
        Audio waveform as float32 numpy array

    Raises:
        ValueError: If events list is empty
    """
    if not events:
        raise ValueError("Events list cannot be empty")

    # Implementation here
    pass
```

### Docstring Format (Google Style)
- Brief one-line summary
- Blank line
- Extended description (if needed)
- Args section with type and description
- Returns section
- Raises section (if applicable)
- Example usage (for complex functions)

### Test Structure
```python
def test_feature_description():
    """Test that feature behaves correctly under normal conditions."""
    # Arrange
    input_data = create_test_data()

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result.shape == expected_shape
    assert np.allclose(result, expected_values, atol=1e-5)
```

## Project Structure

```
physonic/
├── contact/          # Physics & acoustic synthesis
├── spatial/          # Binaural rendering
├── acoustics/        # Room acoustics
├── export/           # Dataset export
└── rl/              # RL environments (future)

tests/
├── test_contact/
├── test_spatial/
├── test_acoustics/
└── test_export/
```

## Documentation

When adding new features:

1. **Update CLAUDE.md** with technical details (internal reference)
2. **Update README.md** with user-facing documentation
3. **Add docstrings** to all public functions/classes
4. **Include examples** in module docstrings

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or design decisions
- Join the community chat (if available)

## Code of Conduct

Be respectful, constructive, and collaborative. We're all here to build better tools for robot perception.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
