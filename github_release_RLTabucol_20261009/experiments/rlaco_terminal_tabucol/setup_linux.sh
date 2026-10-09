#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_ROOT="$PROJECT_ROOT/.venv"

if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
  python -m venv --system-site-packages "$VENV_ROOT"
fi
source "$VENV_ROOT/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$SCRIPT_DIR/requirements-experiment.txt"

if ! command -v cargo >/dev/null 2>&1; then
  echo "Rust/Cargo is missing. Install rustup from https://rustup.rs and rerun this script." >&2
  exit 1
fi

export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
(
  cd "$PROJECT_ROOT/src/tabucol"
  python -m maturin develop --release
)

python "$SCRIPT_DIR/verify_install.py"
echo "Setup complete. Use: $VENV_ROOT/bin/python"
