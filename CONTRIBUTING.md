# Contributing to mankinds-eval

Thank you for your interest in contributing to mankinds-eval. This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git

### Installation

1. Fork and clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/mankinds-eval.git
cd mankinds-eval
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode with all dependencies:

```bash
pip install -e ".[all,dev]"
```

4. Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

The pre-commit hooks will automatically run linting and formatting checks before each commit.

## Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting, and [mypy](https://mypy.readthedocs.io/) for static type checking.

### Linting

```bash
# Check for linting errors
ruff check .

# Auto-fix linting errors where possible
ruff check --fix .
```

### Formatting

```bash
# Check formatting
ruff format --check .

# Apply formatting
ruff format .
```

### Type Checking

```bash
mypy mankinds_eval
```

All code must pass linting, formatting, and type checking before being merged.

## Testing

We use [pytest](https://pytest.org/) for testing.

### Running Tests

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=mankinds_eval --cov-report=term-missing

# Run a specific test file
pytest tests/test_core/test_sample.py

# Run tests matching a pattern
pytest -k "test_fuzzy"
```

### Test Requirements

- All new features must include tests
- All bug fixes should include a regression test
- Aim for high test coverage on new code
- Tests should be fast and deterministic

### Writing Tests

Place tests in the `tests/` directory, mirroring the structure of `mankinds_eval/`:

```
tests/
  test_core/
    test_sample.py
    test_result.py
  test_methods/
    test_heuristic.py
    test_llm.py
```

## Pull Request Process

1. **Create a branch** from `main` for your changes:

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** following the code style guidelines.

3. **Write or update tests** for your changes.

4. **Run the full test suite** to ensure nothing is broken:

```bash
pytest
ruff check .
mypy mankinds_eval
```

5. **Commit your changes** with a clear, descriptive commit message:

```bash
git commit -m "Add support for custom similarity metrics"
```

6. **Push to your fork** and create a pull request.

7. **Fill out the PR description** explaining:
   - What changes were made
   - Why the changes were made
   - How to test the changes

8. **Address review feedback** if requested.

### PR Guidelines

- Keep PRs focused on a single feature or fix
- Update documentation if your changes affect public APIs
- Add a changelog entry for user-facing changes
- Ensure CI passes before requesting review

## Issue Reporting

### Bug Reports

When reporting a bug, please include:

- Python version and operating system
- mankinds-eval version (`pip show mankinds-eval`)
- Minimal code example that reproduces the issue
- Full error message and traceback
- Expected behavior vs actual behavior

### Feature Requests

For feature requests, please describe:

- The problem you are trying to solve
- Your proposed solution
- Alternative approaches you have considered
- Whether you would be willing to implement it

### Before Submitting

- Search existing issues to avoid duplicates
- Use the appropriate issue template if available
- Provide as much context as possible

## Documentation

Documentation is located in the `docs/` directory and built with MkDocs.

### Building Documentation Locally

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.

### Documentation Guidelines

- Update docstrings when changing function signatures
- Add examples for new features
- Keep documentation concise and practical

## Questions

If you have questions about contributing, feel free to open a discussion or issue on GitHub.

## License

By contributing to mankinds-eval, you agree that your contributions will be licensed under the Apache 2.0 License.
