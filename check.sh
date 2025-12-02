#!/usr/bin/env bash
set -euo pipefail

# Always run from the repo root (where pyproject.toml lives)
ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

echo "==> Repo root: $ROOT_DIR"

echo "==> Using Python: $(command -v python)"
python --version

# Name of the virtualenv for docs
DOCS_ENV=".asdfsv"

if [ ! -d "$DOCS_ENV" ]; then
  echo "==> Creating virtualenv in $DOCS_ENV"
  python -m venv "$DOCS_ENV"
fi

# Activate the venv
# shellcheck source=/dev/null
source "$DOCS_ENV/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing deepxde in editable mode"
pip install -e .

echo "==> Installing docs requirements"
pip install -r docs/requirements.txt

echo "==> Building Sphinx docs (like CI/RTD)"
cd docs
python -m sphinx -T -b html -d _build/doctrees -D language=en . _build/html

echo "✅ Docs build succeeded."
