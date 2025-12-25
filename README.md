# TinkexCookbook

<p align="center">
  <img src="assets/tinkex_cookbook.svg" alt="TinkexCookbook Logo" width="200">
</p>

Elixir port of `tinker-cookbook`: training and evaluation recipes for the Tinker ML platform.

## Status: Phase 1 Complete, Phase 2 Part 1 Infrastructure Complete

Phase 1 delivers a working foundation for supervised learning with the `sl_basic` recipe. Phase 2 Part 1 adds the shared infrastructure needed for RL, preference/DPO, and distillation recipes.

### Completed Features

- **Renderers**: Llama3 + Qwen3/DeepSeek/KimiK2/GptOss/RoleColon (tool calling framework included)
- **Datasets**: NoRobots + preference/DPO and distillation dataset builders
- **Training Loops**: Supervised + RL (sync/async) + DPO + on-policy distillation
- **Completers + Checkpointing**: Token/message completers and checkpoint utilities
- **Eval + Utilities**: TinkexGenerate, Eval runner, logtree/trace/display helpers

### Quality Gates (Phase 1 baseline, 2025-12-24)

- 174 tests passing
- Zero compiler warnings
- Credo strict: clean
- Dialyzer: no type errors

### Phase 2 Part 1 Decisions (2025-12-24)

- **VL renderers deferred:** Qwen3VL/Qwen3VLInstruct move to Phase 3; Phase 2 recipes should avoid VL models.
- **Tool calling framework:** Implement shared tool-call encode/decode utilities and reuse them across renderers.
- **Sync + async RL together:** Build sync and async RL training paths in the same pass with shared core abstractions.

See `docs/20251224/phase2_prerequisites/PHASE2_PART1_INFRASTRUCTURE.md` for the updated infrastructure plan.

## Installation

Add `tinkex_cookbook` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:tinkex_cookbook, "~> 0.3.1"}
  ]
end
```

## Quick Start

### Running sl_basic Recipe

```bash
# Set your API key
export TINKER_API_KEY=your_key_here

# Basic run with defaults
mix sl_basic

# With custom options
mix sl_basic log_path=/tmp/my_run learning_rate=0.0002 num_epochs=2

# Limit samples for quick testing
mix sl_basic n_train_samples=100 batch_size=32

# See all available options
mix help sl_basic
```

Available options:
- `log_path` - Output directory (default: `/tmp/tinkex-examples/sl_basic`)
- `model_name` - Model to fine-tune (default: `meta-llama/Llama-3.1-8B`)
- `learning_rate` - Learning rate (default: `0.0002`)
- `num_epochs` - Training epochs (default: `1`)
- `batch_size` - Batch size (default: `128`)
- `max_length` - Max sequence length (default: `32768`)
- `lora_rank` - LoRA rank (default: `32`)
- `n_train_samples` - Limit training samples (default: all)

### Programmatic Usage

```elixir
# Build configuration
config = TinkexCookbook.Recipes.SlBasic.build_config(
  model_name: "meta-llama/Llama-3.1-8B",
  learning_rate: 2.0e-4,
  num_epochs: 1
)

# Run training
TinkexCookbook.Recipes.SlBasic.run_training(config)
```

### Using the Llama3 Renderer

```elixir
alias TinkexCookbook.Renderers.{Llama3, Renderer, TrainOnWhat, Types}

# Initialize renderer with tokenizer
{:ok, state} = Llama3.init(tokenizer: MyTokenizer)

# Build supervised training example
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
```

### Running Evaluations

```elixir
alias TinkexCookbook.Eval.Runner

# Setup config with sampling client
config = %{
  sampling_client: sampling_client,
  model: "meta-llama/Llama-3.1-8B",
  temperature: 0.7,
  max_tokens: 1024,
  stop: ["<|eot_id|>"]
}

# Define samples
samples = [
  %{id: "1", input: "What is 2+2?", target: "4"},
  %{id: "2", input: "What is 3+3?", target: "6"}
]

# Run evaluation
{:ok, results} = Runner.run(samples, config)

# Score and compute metrics
scored = Runner.score_results(results, :exact_match)
metrics = Runner.compute_metrics(scored)
# => %{accuracy: 0.5, total: 2, correct: 1}
```

## Dependencies

Core dependencies for Phase 1:

```elixir
{:tinkex, "~> 0.3.2"}           # Tinker API client
{:chz_ex, "~> 0.1.2"}           # Configuration + CLI
{:hf_datasets_ex, "~> 0.1"}     # HuggingFace datasets
{:crucible_harness, "~> 0.3.1"} # Evaluation framework
{:eval_ex, "~> 0.1.1"}          # Evaluation tasks
{:crucible_datasets, "~> 0.5.1"} # Dataset operations
{:nx, "~> 0.9"}                 # Tensor operations
```

## Documentation

- [AGENTS.md](AGENTS.md) - Agent guide for porting Python to Elixir
- [docs/](docs/) - Architecture and planning documents

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc):

```bash
mix docs
```

## Development

```bash
# Setup
mix deps.get && mix compile

# Tests
mix test

# Code quality
mix format
mix credo --strict
mix dialyzer
```

## License

MIT License - see [LICENSE](LICENSE) for details.
