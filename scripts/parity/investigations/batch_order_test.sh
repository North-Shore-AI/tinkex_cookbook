#!/bin/bash
# Script to compare batch ordering between Python and Elixir
#
# This script extracts and compares the sample content from the first batch
# in both implementations.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PARITY_DIR="/tmp/parity/sl_basic"

echo "=== Batch Order Investigation ==="
echo ""

# Check if parity artifacts exist
if [ ! -d "$PARITY_DIR" ]; then
    echo "Parity artifacts not found. Running parity test first..."
    cd "$REPO_ROOT"
    ./scripts/parity/run_sl_basic_parity.sh || true
fi

echo "=== Python First Batch ==="
if [ -f "$PARITY_DIR/python/parity/first_batch_payload.json" ]; then
    echo "Datum 0 first tokens:"
    jq '.datums[0].model_input.chunks[0:3]' "$PARITY_DIR/python/parity/first_batch_payload.json"
    echo ""
    echo "Datum 0 total length: $(jq '.datums[0].model_input.length' "$PARITY_DIR/python/parity/first_batch_payload.json")"
    echo "Datum 1 total length: $(jq '.datums[1].model_input.length' "$PARITY_DIR/python/parity/first_batch_payload.json")"
fi

echo ""
echo "=== Elixir First Batch ==="
if [ -f "$PARITY_DIR/elixir/parity/first_batch_payload.json" ]; then
    echo "Datum 0 first tokens:"
    jq '.datums[0].model_input.chunks[0:3]' "$PARITY_DIR/elixir/parity/first_batch_payload.json"
    echo ""
    echo "Datum 0 total length: $(jq '.datums[0].model_input.length' "$PARITY_DIR/elixir/parity/first_batch_payload.json")"
    echo "Datum 1 total length: $(jq '.datums[1].model_input.length' "$PARITY_DIR/elixir/parity/first_batch_payload.json")"
fi

echo ""
echo "=== Dataset Sample Order ==="
echo "Python first 4 samples (content hash prefixes):"
jq '.samples[] | "\(.index): \(.content_hash[0:16])..."' "$PARITY_DIR/python/parity/dataset_snapshot.json" 2>/dev/null || echo "Not available"

echo ""
echo "Elixir first 4 samples (content hash prefixes):"
jq '.samples[] | "\(.index): \(.content_hash[0:16])..."' "$PARITY_DIR/elixir/parity/dataset_snapshot.json" 2>/dev/null || echo "Not available"

echo ""
echo "=== Analysis ==="
echo "If the dataset sample order matches but batch content differs,"
echo "the issue is in how samples are assigned to batches after set_epoch()."
echo ""
echo "Python: shuffles HF dataset on set_epoch(), then builds datums lazily"
echo "Elixir: builds datums eagerly, then shuffles datum list on set_epoch()"
echo ""
echo "See INVESTIGATION_REPORT.md for full analysis."
