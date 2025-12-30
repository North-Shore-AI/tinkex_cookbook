# TinkexCookbook

<p align="center">
  <img src="assets/tinkex_cookbook.svg" alt="TinkexCookbook Logo" width="200">
</p>

Elixir port of `tinker-cookbook`: training recipes for the Tinker ML platform, powered by CrucibleKitchen.

## Overview

TinkexCookbook provides ready-to-use training recipes for fine-tuning large language models on the Tinker ML platform. It is built on top of the CrucibleKitchen orchestration engine, providing:

- **Supervised Learning (SFT)** — Fine-tune models on instruction datasets
- **Direct Preference Optimization (DPO)** — Align models with human preferences
- **Reinforcement Learning (CodeRL)** — Train code generation with execution rewards
- Integration with HuggingFace datasets and model hub
- Automatic dataset rendering with Llama3, Qwen3, DeepSeek, and other tokenizers
- Model evaluation with metrics (accuracy, F1, precision, recall)
- Model registration with lineage tracking
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

### 2. Run a Training Workflow

#### Using `mix kitchen.run` (Recommended)

```bash
# Basic supervised training with defaults
mix kitchen.run :supervised

# With custom parameters
mix kitchen.run :supervised \
  --model meta-llama/Llama-3.1-8B \
  --dataset HuggingFaceH4/no_robots \
  --epochs 2 \
  --batch-size 64 \
  --learning-rate 2.0e-4

# DPO training for preference alignment
mix kitchen.run :dpo \
  --model meta-llama/Llama-3.2-1B \
  --dataset hhh \
  --dpo-beta 0.1 \
  --epochs 1

# CodeRL training for code generation
mix kitchen.run :code_rl \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --env deepcoder \
  --group-size 4 \
  --num-rollouts 100

# Dry run to validate configuration
mix kitchen.run :supervised --model my-model --dry-run
```

#### Using the Original Mix Task

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
mix help kitchen.run
mix help sl_basic
```

## Configuration Options

### `mix kitchen.run` Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--model` | `-m` | Model to fine-tune | `meta-llama/Llama-3.1-8B` |
| `--dataset` | `-d` | Dataset name | `HuggingFaceH4/no_robots` |
| `--epochs` | `-e` | Number of training epochs | `1` |
| `--batch-size` | `-b` | Batch size | `128` |
| `--learning-rate` | `-l` | Learning rate | `2.0e-4` |
| `--lora-rank` | `-r` | LoRA rank | `32` |
| `--save-every` | | Checkpoint every N steps | `20` |
| `--eval-every` | | Evaluate every N steps | `0` |
| `--max-length` | | Maximum sequence length | `32768` |
| `--dry-run` | | Validate config without running | `false` |

### `mix sl_basic` Options

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

### Supervised Learning

```elixir
# Run with the V2 recipe using CrucibleKitchen
{:ok, result} = TinkexCookbook.Recipes.SlBasicV2.run(%{
  model: "meta-llama/Llama-3.1-8B",
  dataset: :no_robots,
  epochs: 2,
  learning_rate: 2.0e-4
})

# Access evaluation results
result.context.state.eval_results
# => %{accuracy: 0.92, f1: 0.89, precision: 0.91, recall: 0.88}

# Access registered model
result.context.state.registered_model
# => %{id: "model_123", name: "llama-3.1-8b-sft", version: "1.0.0"}
```

### DPO Training

```elixir
# Direct Preference Optimization for alignment
{:ok, result} = TinkexCookbook.Recipes.DPO.run(%{
  model: "meta-llama/Llama-3.2-1B",
  dataset: :hhh,           # HHH alignment dataset
  dpo_beta: 0.1,           # Preference constraint strength
  epochs: 1,
  batch_size: 256,
  learning_rate: 1.0e-5
})

# Access DPO metrics
result.context.state.dpo_metrics
# => %{loss: 0.32, accuracy: 0.78, margin: 1.2, chosen_reward: 0.8, rejected_reward: -0.4}
```

### CodeRL Training

```elixir
# GRPO-style RL training for code generation
{:ok, result} = TinkexCookbook.Recipes.CodeRL.run(%{
  model: "meta-llama/Llama-3.1-8B-Instruct",
  env: :deepcoder,
  group_size: 4,
  groups_per_batch: 100,
  num_rollouts: 100,
  clip_epsilon: 0.2,
  ppo_epochs: 4,
  learning_rate: 1.0e-5
})

# Access RL metrics
result.context.state.rollout_metrics
# => %{reward_mean: 0.75, reward_std: 0.15, num_trajectories: 400}

result.context.state.ppo_metrics
# => %{policy_loss: 0.02, entropy: 1.2, clip_fraction: 0.12}
```

### Adapter Configuration

```elixir
# Provide custom adapters from crucible_kitchen
adapters = %{
  training_client: {CrucibleKitchen.Adapters.Tinkex.TrainingClient,
    api_key: "custom_key",
    base_url: "https://custom-endpoint.example.com"
  },
  dataset_store: {CrucibleKitchen.Adapters.HfDatasets.DatasetStore, []},
  evaluator: {CrucibleKitchen.Adapters.Noop.Evaluator, []},
  model_registry: {CrucibleKitchen.Adapters.Noop.ModelRegistry, []}
}

CrucibleKitchen.run(TinkexCookbook.Recipes.SlBasicV2, config, adapters: adapters)
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

## Training Pipelines

### Supervised Learning

```
load_dataset → init_session → training_loop → save_final → evaluate → register_model → cleanup
```

1. **Load Dataset** — Fetches training data from HuggingFace
2. **Init Session** — Starts training session with Tinker
3. **Training Loop** — Epochs with forward/backward and optimizer steps
4. **Save Final** — Persists final model weights
5. **Evaluate** — Computes accuracy, F1, precision, recall
6. **Register Model** — Registers model with lineage tracking
7. **Cleanup** — Releases resources

### DPO (Preference Learning)

```
load_dataset → init_session → build_prefs → training_loop → save_final → cleanup
                                                  │
                                         ┌────────┴────────┐
                                         │  For each epoch │
                                         │    └─ batches   │
                                         │       └─ dpo    │
                                         └─────────────────┘
```

1. **Load Dataset** — Fetches preference pairs from HuggingFace
2. **Init Session** — Starts training session with Tinker
3. **Build Preferences** — Constructs preference dataset from raw comparisons
4. **Training Loop** — Computes reference logprobs, DPO loss, and updates policy
5. **Save Final** — Persists final model weights
6. **Cleanup** — Releases resources

### CodeRL (Reinforcement Learning)

```
build_env_group → rollouts_loop → save_final → cleanup
                        │
               ┌────────┴────────┐
               │  For each batch │
               │    └─ rollout   │
               │    └─ advantages│
               │    └─ ppo_update│
               └─────────────────┘
```

1. **Build Env Group** — Creates parallel environment group
2. **Rollout** — Generates code solutions, executes tests, computes rewards
3. **Compute Advantages** — Generalized Advantage Estimation (GAE)
4. **PPO Update** — Policy gradient with clipping
5. **Save Final** — Persists final model weights
6. **Cleanup** — Releases resources

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TINKER_API_KEY` | Yes | Your Tinker platform API key |
| `TINKER_BASE_URL` | No | Custom Tinker API endpoint |
| `HUGGING_FACE_HUB_TOKEN` | No | HuggingFace token for gated models/datasets |

## Available Recipes

### Supervised Learning

| Recipe | Description | CLI |
|--------|-------------|-----|
| `SlBasic` | Original supervised fine-tuning | `mix sl_basic` |
| `SlBasicV2` | CrucibleKitchen-based training with eval & registry | `mix kitchen.run :sl_basic_v2` |
| `:supervised` | Built-in supervised workflow | `mix kitchen.run :supervised` |

### Preference Learning (DPO)

| Recipe | Description | CLI |
|--------|-------------|-----|
| `DPO` | Direct Preference Optimization for alignment | `mix kitchen.run :dpo` |

**Supported datasets:** `:hhh`, `:ultrafeedback`, `:helpsteer3`

**Key parameters:**
- `dpo_beta` — Preference constraint strength (default: 0.1)
- `reference_model` — Frozen reference model (default: same as model)

### Reinforcement Learning

| Recipe | Description | CLI |
|--------|-------------|-----|
| `CodeRL` | Code generation via RL with execution rewards | `mix kitchen.run :code_rl` |

**Supported environments:** `:deepcoder`, `:humaneval` (eval only)

**Key parameters:**
- `group_size` — Parallel rollout group size (default: 4)
- `groups_per_batch` — Groups per training batch (default: 100)
- `clip_epsilon` — PPO clipping parameter (default: 0.2)
- `ppo_epochs` — PPO update epochs (default: 4)
- `gamma` — Discount factor (default: 0.99)
- `gae_lambda` — GAE lambda (default: 0.95)

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

# Run integration tests (requires TINKER_API_KEY)
mix test --include integration

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
| `crucible_model_registry` | Model versioning and lineage |
| `eval_ex` | Model evaluation harness |
| `tinkex` | Elixir client for Tinker API |
| `hf_datasets_ex` | HuggingFace datasets client |
| `hf_hub_ex` | HuggingFace Hub client |

## License

MIT License - see [LICENSE](LICENSE) for details.
