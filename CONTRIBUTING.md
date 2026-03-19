# Contributing to QuantumRiskEngine

Thank you for your interest in contributing to QuantumRiskEngine.

## Development Setup

```bash
git clone https://github.com/charismaforever/QuantumRiskEngine.git
cd QuantumRiskEngine
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## Code Style

This project uses `black` for formatting and `ruff` for linting:

```bash
black .
ruff check .
```

## Adding a New Scenario

1. Create a new YAML file in `config/scenarios/`
2. Follow the schema in `core/scenario.py`
3. Add an integration test in `tests/integration/`

## Module Architecture

Each domain module follows this interface pattern:

```python
class MyModule:
    def __init__(self, params: MyParams, rng: np.random.Generator) -> None: ...
    def compute_score(self) -> float: ...  # Returns [0, 1] scalar for pipeline
```

All modules accept a seeded `np.random.Generator` for full reproducibility.

## Pull Request Guidelines

- Include tests for any new functionality
- Update docstrings and type hints
- Keep PRs focused — one feature/fix per PR
- All CI checks must pass before merge
