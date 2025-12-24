# TinkexCookbook

<p align="center">
  <img src="assets/tinkex_cookbook.svg" alt="TinkexCookbook Logo" width="200">
</p>

Elixir port of `tinker-cookbook`: training and evaluation recipes for the Tinker ML platform.

## Status: Phase 1 Complete

Phase 1 delivers a working foundation for supervised learning with the `sl_basic` recipe.

### Completed Features

- **Llama3 Renderer**: Full implementation of Llama 3 chat template rendering
- **NoRobots Dataset Builder**: Integration with HuggingFace datasets
- **Supervised Training Module**: Training loop orchestration with Tinkex clients
- **TinkexGenerate Adapter**: Bridges CrucibleHarness evaluation to Tinkex sampling
- **Evaluation Runner**: Simple evaluation orchestration with scoring

### Quality Gates

- 165 tests passing
- Zero compiler warnings
- Credo strict: clean
- Dialyzer: no type errors

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
# From command line
mix run -e "TinkexCookbook.Recipes.SlBasic.main()"

# With custom args
mix run -e "TinkexCookbook.Recipes.SlBasic.main()" -- \
  log_path=/tmp/my_run \
  learning_rate=0.0002 \
  num_epochs=2
```

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
