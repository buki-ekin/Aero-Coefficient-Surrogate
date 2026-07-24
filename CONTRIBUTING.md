# Contributing

1. Create a focused branch.
2. Install development dependencies with `pip install -e ".[dev]"`.
3. Add or update tests with every behavior change.
4. Run:

   ```bash
   ruff check src tests
   pytest --cov=aero_surrogate --cov-fail-under=75
   python -m build
   ```

5. Keep commits small, descriptive, and scientifically traceable.
6. Update documentation and `CHANGELOG.md` when user-visible behavior changes.

Do not commit generated secrets, untrusted model files, or results whose data
source cannot be documented.
