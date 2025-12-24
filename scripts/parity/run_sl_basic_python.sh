#!/bin/bash
#
# Run Python sl_basic recipe for parity comparison.
#
# This script runs the Python sl_basic recipe with parity mode enabled
# to collect artifacts for comparison with the Elixir implementation.
#
# Usage (from repo root):
#   ./scripts/parity/run_sl_basic_python.sh
#
# Environment:
#   TINKER_API_KEY - Required. Your Tinker API key.
#   TINKER_BASE_URL - Optional. Tinker API base URL.
#
# Output:
#   Artifacts are written to /tmp/parity/sl_basic/python/
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_COOKBOOK="$REPO_ROOT/tinker-cookbook"
VENV_DIR="$PYTHON_COOKBOOK/.venv"

# Default output directory
PARITY_OUTPUT_DIR="${PARITY_OUTPUT_DIR:-/tmp/parity/sl_basic/python}"

# Check for required environment
if [ -z "$TINKER_API_KEY" ]; then
    echo "Error: TINKER_API_KEY environment variable is required"
    exit 1
fi

echo "=== Python sl_basic Parity Run ==="
echo "Output directory: $PARITY_OUTPUT_DIR"
echo "Repository root: $REPO_ROOT"
echo ""

# Setup Python virtual environment if needed
setup_venv() {
    cd "$PYTHON_COOKBOOK"

    # Check if venv exists AND has chz installed (key dependency)
    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python" ]; then
        if "$VENV_DIR/bin/python" -c "import chz" 2>/dev/null; then
            echo "Using existing venv at $VENV_DIR"
            cd "$REPO_ROOT"
            return 0
        else
            echo "Venv exists but dependencies incomplete. Reinstalling..."
        fi
    else
        echo "Setting up Python virtual environment..."
    fi

    # Try uv first (faster), fall back to python venv + pip
    if command -v uv &> /dev/null; then
        echo "Using uv to create venv and install dependencies..."
        [ ! -d "$VENV_DIR" ] && uv venv "$VENV_DIR"
        echo "Installing dependencies (this may take a few minutes on first run)..."
        uv pip install -e .
    else
        echo "Using python3 venv + pip to install dependencies..."
        [ ! -d "$VENV_DIR" ] && python3 -m venv "$VENV_DIR"
        "$VENV_DIR/bin/pip" install --upgrade pip
        echo "Installing dependencies (this may take a few minutes on first run)..."
        "$VENV_DIR/bin/pip" install -e .
    fi

    echo "Python environment ready."
    cd "$REPO_ROOT"
}

# Setup venv
setup_venv

# Activate venv
source "$VENV_DIR/bin/activate"

# Clean output directory completely so Python doesn't prompt
rm -rf "$PARITY_OUTPUT_DIR"

# Run Python sl_basic with parity mode enabled
cd "$REPO_ROOT"

# Use parity-specific entry point that supports sample limiting
PYTHONPATH="$PYTHON_COOKBOOK:$PYTHONPATH" \
PARITY_MODE=1 \
python -m tinker_cookbook.recipes.sl_basic_parity \
    log_path="$PARITY_OUTPUT_DIR" \
    batch_size=2 \
    num_epochs=1 \
    max_length=256 \
    n_train_samples=4

echo ""
echo "=== Python run complete ==="
echo "Artifacts saved to: $PARITY_OUTPUT_DIR"
echo ""

# List artifacts
echo "Artifacts:"
ls -la "$PARITY_OUTPUT_DIR/"
if [ -d "$PARITY_OUTPUT_DIR/parity" ]; then
    echo ""
    echo "Parity artifacts:"
    ls -la "$PARITY_OUTPUT_DIR/parity/"
fi
