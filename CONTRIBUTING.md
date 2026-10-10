# Contributing to NeuroGraph-LLM

We welcome contributions to the NeuroGraph-LLM research prototype! Please review our contribution guidelines before submitting code.

## Development Workflow

1. Fork and clone the repository.
2. Ensure you have Python 3.10+ (preferably Python 3.11).
3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Run the test suite:
   ```bash
   pytest -q
   ```

## Code Quality Standards

- Maintain complete type hints and docstrings.
- Code style: adhere to standard Python formatting (ruff / black compatible).
- All new architectural modules must include dedicated unit tests in `tests/`.
- No large model weights or binary blobs should be committed to the repository.
- Avoid external proprietary API dependencies.

## Scientific Rigor

- Negative results are first-class contributions. If an architectural variation underperforms, report the configuration and empirical findings without exaggeration.
- Claims must be supported by reproducible experiments and executed benchmark logs.
