#!/bin/bash
set -e

# Comprehensive Release Quality Verification Script
# Automates the "5-minute ritual" to ensure zero point-of-contact failures.

echo "=========================================================="
echo "          dspyer Pre-Release Validation Pipeline          "
echo "=========================================================="

# 1. Run local link checker
echo -e "\n[1/5] Running link verifier..."
python3 scripts/verify_links.py

# 2. Run local code quality gates
echo -e "\n[2/5] Running style & type checks..."
uv run ruff check dspyer tests examples
uv run ruff format --check dspyer tests examples
uv run mypy dspyer tests examples

# 3. Run full unit test suite
echo -e "\n[3/5] Running pytest suite..."
uv run pytest

# 4. Build documentation site
echo -e "\n[4/5] Testing documentation build..."
uv run mkdocs build

# 5. Build package and run isolated smoke test
echo -e "\n[5/5] Building package and running isolated smoke test..."

# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build wheel
uv build

# Create isolated temporary virtual environment
TEMP_VENV=$(mktemp -d)/smoke_venv
echo "Creating isolated virtual environment at $TEMP_VENV..."
python3 -m venv "$TEMP_VENV"

# Install built wheel in isolated environment
echo "Installing built wheel..."
"$TEMP_VENV/bin/pip" install --upgrade pip
"$TEMP_VENV/bin/pip" install dist/*.whl dspy-ai pydantic

# Run smoke test in isolation (forcing PYTHONPATH to be empty to prevent importing local source)
echo "Running smoke test inside isolated environment..."
PYTHONPATH="" "$TEMP_VENV/bin/python" scripts/smoke_test.py

# Clean up
echo "Cleaning up temporary environment..."
rm -rf "$TEMP_VENV"

echo -e "\n=========================================================="
echo "  [SUCCESS] All checks passed! Package is 100% release-ready."
echo "=========================================================="
