#!/bin/bash
#
# Run isolated parity test cases to identify specific library-level deviations.
#
# This script runs the isolated test cases documented in:
#   docs/20251224/parity_investigation/PARITY_MISMATCH_INVESTIGATION.md
#
# Usage (from repo root):
#   ./scripts/parity/run_isolation_tests.sh [test_number]
#
# Tests:
#   1 - Dataset ordering (HuggingFace datasets vs HfDatasetsEx)
#   2 - Renderer token parity (Llama3Renderer)
#   3 - Tokenizer parity (AutoTokenizer vs Tinkex.Tokenizer)
#   4 - Datum construction parity
#   all - Run all tests (default)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_COOKBOOK="$REPO_ROOT/tinker-cookbook"
VENV_DIR="$PYTHON_COOKBOOK/.venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_test() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

echo_pass() {
    echo -e "${GREEN}PASS: $1${NC}"
}

echo_fail() {
    echo -e "${RED}FAIL: $1${NC}"
}

echo_info() {
    echo -e "${YELLOW}INFO: $1${NC}"
}

# Setup Python venv if needed
setup_python() {
    if [ ! -d "$VENV_DIR" ] || ! "$VENV_DIR/bin/python" -c "import datasets, transformers" 2>/dev/null; then
        echo_info "Setting up Python environment..."
        cd "$PYTHON_COOKBOOK"
        if command -v uv &> /dev/null; then
            [ ! -d "$VENV_DIR" ] && uv venv "$VENV_DIR"
            uv pip install -e .
        else
            [ ! -d "$VENV_DIR" ] && python3 -m venv "$VENV_DIR"
            "$VENV_DIR/bin/pip" install -e .
        fi
        cd "$REPO_ROOT"
    fi
}

# Test 1: Dataset Ordering
test_dataset_ordering() {
    echo_test "Test 1: Dataset Ordering (HuggingFace datasets vs HfDatasetsEx)"
    echo ""

    # Python test
    echo "--- Python: Loading no_robots dataset ---"
    cd "$PYTHON_COOKBOOK"
    "$VENV_DIR/bin/python" -c "
import datasets
import hashlib

ds = datasets.load_dataset('HuggingFaceH4/no_robots')['train']

print('=== Python: WITHOUT shuffle (original order) ===')
for i in range(4):
    content = ds[i]['messages'][0]['content']
    h = hashlib.sha256(content.encode()).hexdigest()[:16]
    print(f'Sample {i}: hash={h} | {content[:60]}...')

ds_shuffled = ds.shuffle(seed=0)
print()
print('=== Python: WITH shuffle(seed=0) ===')
for i in range(4):
    content = ds_shuffled[i]['messages'][0]['content']
    h = hashlib.sha256(content.encode()).hexdigest()[:16]
    print(f'Sample {i}: hash={h} | {content[:60]}...')
"

    echo ""

    # Elixir test
    echo "--- Elixir: Loading no_robots dataset ---"
    cd "$REPO_ROOT"
    mix run -e '
{:ok, dataset} = HfDatasetsEx.load_dataset("HuggingFaceH4/no_robots", split: "train")

IO.puts("=== Elixir: WITHOUT shuffle (original order) ===")
dataset.items
|> Enum.take(4)
|> Enum.with_index(fn sample, i ->
    content = sample["messages"] |> hd() |> Map.get("content")
    h = :crypto.hash(:sha256, content) |> Base.encode16(case: :lower) |> String.slice(0, 16)
    preview = String.slice(content, 0, 60)
    IO.puts("Sample #{i}: hash=#{h} | #{preview}...")
end)

# Now test with numpy-compatible shuffle (PCG64)
shuffled = HfDatasetsEx.Dataset.shuffle(dataset, seed: 0)

IO.puts("")
IO.puts("=== Elixir: WITH shuffle(seed=0) using PCG64 ===")
shuffled.items
|> Enum.take(4)
|> Enum.with_index(fn sample, i ->
    content = sample["messages"] |> hd() |> Map.get("content")
    h = :crypto.hash(:sha256, content) |> Base.encode16(case: :lower) |> String.slice(0, 16)
    preview = String.slice(content, 0, 60)
    IO.puts("Sample #{i}: hash=#{h} | #{preview}...")
end)
'

    echo ""
    echo_info "EXPECTED: Python WITHOUT shuffle should match Elixir WITHOUT shuffle"
    echo_info "EXPECTED: Python WITH shuffle(seed=0) should match Elixir WITH shuffle(seed=0)"
}

# Test 2: Renderer Token Parity
test_renderer_parity() {
    echo_test "Test 2: Renderer Token Parity (Llama3Renderer)"
    echo ""

    # Python test
    echo "--- Python: Rendering test message ---"
    cd "$PYTHON_COOKBOOK"
    "$VENV_DIR/bin/python" -c "
from transformers import AutoTokenizer
from tinker_cookbook.renderers import Llama3Renderer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
renderer = Llama3Renderer(tokenizer)

message = {'role': 'user', 'content': 'Hello, how are you today?'}
rendered = renderer.render_message(0, message)

print('=== Python Llama3Renderer ===')
print(f'BOS tokens: {renderer._bos_tokens}')
print(f'Prefix tokens: {list(rendered.prefix.tokens)}')
print(f'Content tokens: {list(rendered.content[0].tokens)}')
print(f'Stop sequences: {renderer.get_stop_sequences()}')
"

    echo ""

    # Elixir test
    echo "--- Elixir: Rendering test message ---"
    cd "$REPO_ROOT"
    mix run -e '
alias TinkexCookbook.Renderers.Llama3
alias Tinkex.Tokenizer

{:ok, tokenizer} = Tokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
{:ok, state} = Llama3.init(tokenizer: tokenizer)

message = TinkexCookbook.Renderers.Types.message("user", "Hello, how are you today?")
{rendered, _state} = Llama3.render_message(0, message, false, state)

IO.puts("=== Elixir Llama3 Renderer ===")
IO.puts("BOS tokens: #{inspect(Llama3.bos_tokens(state))}")
IO.puts("Prefix tokens: #{inspect(rendered.prefix.tokens)}")
IO.puts("Content tokens: #{inspect(hd(rendered.content).tokens)}")
IO.puts("Stop sequences: #{inspect(Llama3.stop_sequences(state))}")
'

    echo ""
    echo_info "EXPECTED: All token sequences should be IDENTICAL"
}

# Test 3: Tokenizer Parity
test_tokenizer_parity() {
    echo_test "Test 3: Tokenizer Parity"
    echo ""

    # Python test
    echo "--- Python: Tokenizing test strings ---"
    cd "$PYTHON_COOKBOOK"
    "$VENV_DIR/bin/python" -c "
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')

test_strings = [
    'Hello, world!',
    'The quick brown fox jumps over the lazy dog.',
    '123 + 456 = 579',
    '<|start_header_id|>user<|end_header_id|>',
    'Line 1\n\nLine 3',
]

print('=== Python AutoTokenizer ===')
for s in test_strings:
    tokens = tokenizer.encode(s, add_special_tokens=False)
    print(f'{repr(s)[:50]:50s} -> {tokens}')
"

    echo ""

    # Elixir test
    echo "--- Elixir: Tokenizing test strings ---"
    cd "$REPO_ROOT"
    mix run -e '
alias Tinkex.Tokenizer

{:ok, tokenizer} = Tokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

test_strings = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "123 + 456 = 579",
    "<|start_header_id|>user<|end_header_id|>",
    "Line 1\n\nLine 3",
]

IO.puts("=== Elixir Tinkex.Tokenizer ===")
Enum.each(test_strings, fn s ->
    tokens = Tokenizer.encode(tokenizer, s, add_special_tokens: false)
    preview = s |> inspect() |> String.slice(0, 50) |> String.pad_trailing(50)
    IO.puts("#{preview} -> #{inspect(tokens)}")
end)
'

    echo ""
    echo_info "EXPECTED: All token sequences should be IDENTICAL"
}

# Test 4: Datum Construction Parity
test_datum_parity() {
    echo_test "Test 4: Datum Construction Parity"
    echo ""

    # Python test
    echo "--- Python: Building datum ---"
    cd "$PYTHON_COOKBOOK"
    "$VENV_DIR/bin/python" -c "
from transformers import AutoTokenizer
from tinker_cookbook.renderers import Llama3Renderer, TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
renderer = Llama3Renderer(tokenizer)

messages = [
    {'role': 'user', 'content': 'What is 2+2?'},
    {'role': 'assistant', 'content': 'The answer is 4.'}
]

datum = conversation_to_datum(
    messages,
    renderer,
    max_length=256,
    train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
)

target_tokens = list(datum.loss_fn_inputs['target_tokens'].data)
weights = list(datum.loss_fn_inputs['weights'].data)

print('=== Python Datum ===')
print(f'Model input length: {datum.model_input.length}')
print(f'Target tokens length: {len(target_tokens)}')
print(f'First 15 target tokens: {target_tokens[:15]}')
print(f'Last 15 target tokens: {target_tokens[-15:]}')
print(f'Weights sum (trainable tokens): {sum(weights)}')
print(f'First 15 weights: {weights[:15]}')
print(f'Last 15 weights: {weights[-15:]}')
"

    echo ""

    # Elixir test
    echo "--- Elixir: Building datum ---"
    cd "$REPO_ROOT"
    mix run -e '
alias TinkexCookbook.Renderers.{Llama3, Renderer, TrainOnWhat, Types}
alias TinkexCookbook.Supervised.Common
alias Tinkex.Tokenizer

{:ok, tokenizer} = Tokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
{:ok, state} = Llama3.init(tokenizer: tokenizer)

messages = [
    Types.message("user", "What is 2+2?"),
    Types.message("assistant", "The answer is 4.")
]

{model_input, weights} = Renderer.build_supervised_example(
    Llama3,
    messages,
    TrainOnWhat.all_assistant_messages(),
    state
)

datum = Common.datum_from_model_input_weights(model_input, weights, 256)

target_tokens = datum.loss_fn_inputs.target_tokens.data
datum_weights = datum.loss_fn_inputs.weights.data

IO.puts("=== Elixir Datum ===")
IO.puts("Model input length: #{datum.model_input.length}")
IO.puts("Target tokens length: #{length(target_tokens)}")
IO.puts("First 15 target tokens: #{inspect(Enum.take(target_tokens, 15))}")
IO.puts("Last 15 target tokens: #{inspect(Enum.take(target_tokens, -15))}")
IO.puts("Weights sum (trainable tokens): #{Enum.sum(datum_weights)}")
IO.puts("First 15 weights: #{inspect(Enum.take(datum_weights, 15))}")
IO.puts("Last 15 weights: #{inspect(Enum.take(datum_weights, -15))}")
'

    echo ""
    echo_info "EXPECTED: All values should be IDENTICAL"
}

# Main
main() {
    cd "$REPO_ROOT"

    echo_test "Parity Isolation Tests"
    echo "Repository: $REPO_ROOT"
    echo ""

    # Setup Python environment
    setup_python

    TEST="${1:-all}"

    case "$TEST" in
        1)
            test_dataset_ordering
            ;;
        2)
            test_renderer_parity
            ;;
        3)
            test_tokenizer_parity
            ;;
        4)
            test_datum_parity
            ;;
        all)
            test_dataset_ordering
            echo ""
            echo "========================================="
            echo ""
            test_renderer_parity
            echo ""
            echo "========================================="
            echo ""
            test_tokenizer_parity
            echo ""
            echo "========================================="
            echo ""
            test_datum_parity
            ;;
        *)
            echo "Unknown test: $TEST"
            echo "Usage: $0 [1|2|3|4|all]"
            exit 1
            ;;
    esac

    echo ""
    echo_test "Tests Complete"
}

main "$@"
