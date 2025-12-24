#!/bin/bash
#
# Run Elixir sl_basic recipe for parity comparison.
#
# This script runs the Elixir sl_basic recipe with parity mode enabled
# to collect artifacts for comparison with the Python implementation.
#
# Usage (from repo root):
#   ./scripts/parity/run_sl_basic_elixir.sh
#
# Environment:
#   TINKER_API_KEY - Required. Your Tinker API key.
#   TINKER_BASE_URL - Optional. Tinker API base URL.
#   TINKEX_HTTP_PROTOCOL - Set to http1 by default to avoid HTTP/2 window errors.
#
# Output:
#   Artifacts are written to /tmp/parity/sl_basic/elixir/
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default output directory
PARITY_OUTPUT_DIR="${PARITY_OUTPUT_DIR:-/tmp/parity/sl_basic/elixir}"

# Check for required environment
if [ -z "$TINKER_API_KEY" ]; then
    echo "Error: TINKER_API_KEY environment variable is required"
    exit 1
fi

echo "=== Elixir sl_basic Parity Run ==="
echo "Output directory: $PARITY_OUTPUT_DIR"
echo "Repository root: $REPO_ROOT"
echo ""

# Clean output directory completely so Elixir doesn't prompt
rm -rf "$PARITY_OUTPUT_DIR"

# Run Elixir sl_basic with parity mode enabled
cd "$REPO_ROOT"

PARITY_MODE=1 \
TINKEX_HTTP_PROTOCOL="${TINKEX_HTTP_PROTOCOL:-http1}" \
mix run -e "TinkexCookbook.Recipes.SlBasic.main()" -- \
    log_path="$PARITY_OUTPUT_DIR" \
    batch_size=2 \
    num_epochs=1 \
    max_length=256 \
    n_train_samples=4

echo ""
echo "=== Elixir run complete ==="
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
