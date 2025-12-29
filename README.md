# TinkexCookbook

<p align="center">
  <img src="assets/tinkex_cookbook.svg" alt="TinkexCookbook Logo" width="200">
</p>

Elixir port of `tinker-cookbook`: training recipes for the Tinker ML platform, powered by CrucibleKitchen.

## Overview

TinkexCookbook provides ready-to-use training recipes for fine-tuning large language models on the Tinker ML platform. It is built on top of the CrucibleKitchen orchestration engine, providing:

- Pre-built recipes for supervised learning, RL, and DPO training
- Integration with HuggingFace datasets and model hub
- Automatic dataset rendering with Llama3, Qwen3, DeepSeek, and other tokenizers
- Full telemetry and checkpoint management

## Prerequisites

- Elixir 1.18 or later
- A Tinker API key (set as `TINKER_API_KEY` environment variable)
- Optional: HuggingFace token for gated models (`HUGGING_FACE_HUB_TOKEN`)

## Installation

Add `tinkex_cookbook` to your dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:tinkex_cookbook, "~> 0.4.0"}
  ]
end
```

Then fetch dependencies:

```bash
mix deps.get
```

## Quick Start

### 1. Set Your API Key

```bash
export TINKER_API_KEY=your_api_key_here
```

### 2. Run the Supervised Learning Recipe

```bash
# Basic run with defaults (Llama-3.1-8B, NoRobots dataset)
mix sl_basic

# With custom parameters
mix sl_basic model_name=meta-llama/Llama-3.1-8B learning_rate=0.0002 num_epochs=2

# Quick test with limited samples
mix sl_basic n_train_samples=100 batch_size=32
```

### 3. View Available Options

```bash
mix help sl_basic
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `model_name` | Model to fine-tune | `meta-llama/Llama-3.1-8B` |
| `learning_rate` | Learning rate | `0.0002` |
| `lr_schedule` | Schedule: `linear`, `constant`, `cosine` | `linear` |
| `num_epochs` | Number of training epochs | `1` |
| `batch_size` | Batch size | `128` |
| `max_length` | Maximum sequence length | `32768` |
| `lora_rank` | LoRA rank | `32` |
| `n_train_samples` | Limit training samples | all |
| `save_every` | Save checkpoint every N steps | `20` |
| `log_path` | Output directory | `/tmp/tinkex-examples/sl_basic` |

## Programmatic Usage

### Using CrucibleKitchen (Recommended)

```elixir
# Run with the V2 recipe using CrucibleKitchen
TinkexCookbook.Recipes.SlBasicV2.run(%{
  model: "meta-llama/Llama-3.1-8B",
  dataset: :no_robots,
  epochs: 2,
  learning_rate: 2.0e-4
})
```

### Direct Recipe Execution

```elixir
# Using the original recipe implementation
config = %{
  model: "meta-llama/Llama-3.1-8B",
  num_epochs: 1,
  batch_size: 128
}

TinkexCookbook.Recipes.SlBasic.run_training(config)
```

### Adapter Configuration

```elixir
# Provide adapters from crucible_kitchen
adapters = %{
  training_client: {CrucibleKitchen.Adapters.Tinkex.TrainingClient,
    api_key: "custom_key",
    base_url: "https://custom-endpoint.example.com"
  },
  dataset_store: {CrucibleKitchen.Adapters.HfDatasets.DatasetStore, []}
}

CrucibleKitchen.run(TinkexCookbook.Recipes.SlBasicV2, config, adapters: adapters)
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TINKER_API_KEY` | Yes | Your Tinker platform API key |
| `TINKER_BASE_URL` | No | Custom Tinker API endpoint |
| `HUGGING_FACE_HUB_TOKEN` | No | HuggingFace token for gated models/datasets |

## Available Recipes

| Recipe | Description | Mix Task |
|--------|-------------|----------|
| `SlBasic` | Supervised fine-tuning with NoRobots | `mix sl_basic` |
| `SlBasicV2` | CrucibleKitchen-based supervised training | Programmatic only |

## Supported Models

Renderers are provided by `crucible_train`:

- **Llama 3.x** family (Llama-3.1-8B, Llama-3.2-3B, etc.)
- **Qwen 3.x** family
- **DeepSeek V3**
- **Generic role-colon format** (fallback)

Renderers handle proper tokenization, special tokens, and training weight computation for each model family.

## Development

```bash
# Setup
mix deps.get && mix compile

# Run tests
mix test

# Code quality
mix format
mix credo --strict
mix dialyzer
```

## Documentation

For internal architecture details, see:

- [Internal Architecture Guide](docs/guides/internal_architecture.md) - CrucibleKitchen integration, adapters, and workflows
- [DEVELOPERS.md](docs/DEVELOPERS.md) - Development guidelines

Generate API documentation:

```bash
mix docs
open doc/index.html
```

## Related Projects

TinkexCookbook is part of the North-Shore-AI ecosystem:

| Project | Purpose |
|---------|---------|
| `crucible_kitchen` | Backend-agnostic training orchestration |
| `crucible_train` | Training types, renderers, and dataset utilities |
| `crucible_ir` | Experiment intermediate representation |
| `tinkex` | Elixir client for Tinker API |
| `hf_datasets_ex` | HuggingFace datasets client |
| `hf_hub_ex` | HuggingFace Hub client |

## License

MIT License - see [LICENSE](LICENSE) for details.
