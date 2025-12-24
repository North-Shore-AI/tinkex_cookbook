#!/bin/bash
#
# Run full sl_basic parity comparison between Python and Elixir.
#
# This script runs both Python and Elixir sl_basic recipes with parity mode
# enabled, then compares the artifacts and produces a summary report.
#
# Usage (from repo root):
#   ./scripts/parity/run_sl_basic_parity.sh
#
# Environment:
#   TINKER_API_KEY - Required. Your Tinker API key.
#   TINKER_BASE_URL - Optional. Tinker API base URL.
#   SKIP_PYTHON - Set to 1 to skip Python run (use existing artifacts).
#   SKIP_ELIXIR - Set to 1 to skip Elixir run (use existing artifacts).
#
# Output:
#   Artifacts are written to /tmp/parity/sl_basic/{python,elixir}/
#   Comparison report is written to /tmp/parity/sl_basic/report.json
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Output directories
PARITY_BASE_DIR="${PARITY_BASE_DIR:-/tmp/parity/sl_basic}"
PYTHON_OUTPUT_DIR="$PARITY_BASE_DIR/python"
ELIXIR_OUTPUT_DIR="$PARITY_BASE_DIR/elixir"

echo "========================================"
echo "   sl_basic Parity Comparison Suite    "
echo "========================================"
echo ""
echo "Output directory: $PARITY_BASE_DIR"
echo ""

# Check for required environment
if [ -z "$TINKER_API_KEY" ]; then
    echo "Error: TINKER_API_KEY environment variable is required"
    exit 1
fi

cd "$REPO_ROOT"

# Run Python if not skipped
if [ "${SKIP_PYTHON:-0}" != "1" ]; then
    echo ""
    echo "========================================"
    echo "Step 1: Running Python sl_basic"
    echo "========================================"
    PARITY_OUTPUT_DIR="$PYTHON_OUTPUT_DIR" ./scripts/parity/run_sl_basic_python.sh
else
    echo ""
    echo "Skipping Python run (SKIP_PYTHON=1)"
fi

# Run Elixir if not skipped
if [ "${SKIP_ELIXIR:-0}" != "1" ]; then
    echo ""
    echo "========================================"
    echo "Step 2: Running Elixir sl_basic"
    echo "========================================"
    PARITY_OUTPUT_DIR="$ELIXIR_OUTPUT_DIR" ./scripts/parity/run_sl_basic_elixir.sh
else
    echo ""
    echo "Skipping Elixir run (SKIP_ELIXIR=1)"
fi

# Run comparison
echo ""
echo "========================================"
echo "Step 3: Comparing Artifacts"
echo "========================================"
echo ""

python3 "$SCRIPT_DIR/compare_artifacts.py" \
    "$PYTHON_OUTPUT_DIR" \
    "$ELIXIR_OUTPUT_DIR" \
    --output "$PARITY_BASE_DIR/report.json"

echo ""
echo "========================================"
echo "Parity comparison complete!"
echo "========================================"
echo ""
echo "Artifacts:"
echo "  Python:  $PYTHON_OUTPUT_DIR"
echo "  Elixir:  $ELIXIR_OUTPUT_DIR"
echo "  Report:  $PARITY_BASE_DIR/report.json"
