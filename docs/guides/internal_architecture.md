# TinkexCookbook Internal Architecture

This guide provides comprehensive technical documentation for developers working on or extending TinkexCookbook.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [CrucibleKitchen Integration](#cruciblekitchen-integration)
3. [Hexagonal Architecture and Adapters](#hexagonal-architecture-and-adapters)
4. [Workflow Engine](#workflow-engine)
5. [Data Flow](#data-flow)
6. [HuggingFace Integration](#huggingface-integration)
7. [Tinkex SDK Integration](#tinkex-sdk-integration)
8. [Snakepit Python Bridge](#snakepit-python-bridge)
9. [Creating New Recipes](#creating-new-recipes)
10. [Configuration Patterns](#configuration-patterns)

---

## Architecture Overview

TinkexCookbook follows a layered architecture where the cookbook provides thin configuration and recipes on top of the CrucibleKitchen orchestration engine. Adapter implementations live in `crucible_kitchen`.

```
+------------------------------------------------------------------+
|                     TINKEX_COOKBOOK                               |
|  Thin frontend: Recipes (config) + runtime glue                   |
+------------------------------------------------------------------+
                               |
                               v
+------------------------------------------------------------------+
|                    CRUCIBLE_KITCHEN                               |
|  Backend-agnostic orchestration engine                           |
|                                                                   |
|  +---------------+  +---------------+  +---------------+         |
|  |    Recipes    |  |   Workflows   |  |   Telemetry   |         |
|  |  (SL/RL/DPO)  |  |  (Pipelines)  |  |   (Metrics)   |         |
|  +---------------+  +---------------+  +---------------+         |
|  +---------------+  +---------------+  +---------------+         |
|  |     Ports     |  |    Stages     |  |    Context    |         |
|  |  (Contracts)  |  | (Operations)  |  |   (State)     |         |
|  +---------------+  +---------------+  +---------------+         |
+------------------------------------------------------------------+
                               |
          +--------------------+--------------------+
          v                    v                    v
+------------------+  +------------------+  +------------------+
|  CRUCIBLE_TRAIN  |  | CRUCIBLE_FRAME-  |  |   CRUCIBLE_IR    |
|                  |  |     WORK         |  |                  |
| Types, Renderers |  | Pipeline Runner  |  | Experiment Specs |
+------------------+  +------------------+  +------------------+
```

### Design Principles

1. **Inversion of Control**: Cookbooks don't call the kitchen; the kitchen uses configured adapters from `crucible_kitchen`
2. **Workflow-Centric**: Composable sequences of operations with lifecycle management
3. **Configuration-Driven**: Recipes are mostly configuration; the kitchen does the work
4. **Observable by Default**: Every operation emits telemetry

---

## CrucibleKitchen Integration

### Recipe Behaviour

TinkexCookbook recipes implement the `CrucibleKitchen.Recipe` behaviour via the `use CrucibleKitchen.Recipe` macro:

```elixir
defmodule TinkexCookbook.Recipes.SlBasicV2 do
  use CrucibleKitchen.Recipe

  @impl true
  def name, do: :sl_basic_v2

  @impl true
  def description do
    "Supervised fine-tuning using CrucibleKitchen with Tinkex backend"
  end

  @impl true
  def default_config do
    %{
      model: "meta-llama/Llama-3.1-8B",
      lora_rank: 32,
      dataset: :no_robots,
      epochs: 1,
      batch_size: 128,
      learning_rate: 2.0e-4
    }
  end

  @impl true
  def required_adapters do
    [:training_client, :dataset_store]
  end

  @impl true
  def workflow do
    CrucibleKitchen.Workflows.Supervised.__workflow__()
  end

  @impl true
  def validate_config(config) do
    # Validation logic
    :ok
  end
end
```

### Running Recipes

Recipes are executed through `CrucibleKitchen.run/3`:

```elixir
# Full form
adapters = %{
  training_client: {CrucibleKitchen.Adapters.Tinkex.TrainingClient, []},
  dataset_store: {CrucibleKitchen.Adapters.HfDatasets.DatasetStore, []}
}

CrucibleKitchen.run(TinkexCookbook.Recipes.SlBasicV2, config, adapters: adapters)

# Convenience wrapper
TinkexCookbook.Recipes.SlBasicV2.run(config)
```

---

## Hexagonal Architecture and Adapters

CrucibleKitchen uses ports and adapters to abstract external dependencies; TinkexCookbook recipes supply configuration for those adapters.

### Port Definitions

Ports are behaviour contracts defined in CrucibleTrain:

```
+-------------------+     +-------------------+     +-------------------+
|  TrainingClient   |     |   DatasetStore    |     |    HubClient      |
|       Port        |     |       Port        |     |       Port        |
+-------------------+     +-------------------+     +-------------------+
         |                         |                         |
         v                         v                         v
+-------------------+     +-------------------+     +-------------------+
|  Tinkex Adapter   |     | HfDatasets Adapter|     |  HfHub Adapter    |
+-------------------+     +-------------------+     +-------------------+
         |                         |                         |
         v                         v                         v
+-------------------+     +-------------------+     +-------------------+
|   Tinkex SDK      |     |  HfDatasetsEx     |     |     HfHub         |
+-------------------+     +-------------------+     +-------------------+
```

### Available Adapters (in crucible_kitchen)

#### TrainingClient Adapter (Tinkex)

Location: `crucible_kitchen/lib/crucible_kitchen/adapters/tinkex/training_client.ex`

Implements `CrucibleTrain.Ports.TrainingClient`:

```elixir
defmodule CrucibleKitchen.Adapters.Tinkex.TrainingClient do
  @behaviour CrucibleTrain.Ports.TrainingClient

  @impl true
  def start_session(adapter_opts, config) do
    # Creates Tinkex.ServiceClient and Tinkex.TrainingClient
  end

  @impl true
  def forward_backward(_adapter_opts, session, datums) do
    # Converts CrucibleTrain.Types.Datum to Tinkex format
    # Returns async future
  end

  @impl true
  def optim_step(_adapter_opts, session, lr) do
    # Adam optimizer step with learning rate
  end

  @impl true
  def save_checkpoint(_adapter_opts, session, name) do
    # Saves training state
  end

  @impl true
  def close_session(_adapter_opts, session) do
    # Cleanup
  end
end
```

#### DatasetStore Adapter (HfDatasets)

Location: `crucible_kitchen/lib/crucible_kitchen/adapters/hf_datasets/dataset_store.ex`

Implements `CrucibleTrain.Ports.DatasetStore`:

```elixir
defmodule CrucibleKitchen.Adapters.HfDatasets.DatasetStore do
  @behaviour CrucibleTrain.Ports.DatasetStore

  @impl true
  def load_dataset(_opts, repo_id, opts) do
    HfDatasetsEx.load_dataset(repo_id, opts)
  end

  @impl true
  def get_split(_opts, dataset_dict, split) do
    HfDatasetsEx.DatasetDict.get(dataset_dict, split)
  end

  @impl true
  def shuffle(_opts, dataset, opts) do
    HfDatasetsEx.Dataset.shuffle(dataset, opts)
  end

  @impl true
  def to_list(_opts, dataset) do
    HfDatasetsEx.Dataset.to_list(dataset)
  end
end
```

#### HubClient Adapter (HfHub)

Location: `crucible_kitchen/lib/crucible_kitchen/adapters/hf_hub/hub_client.ex`

Implements `CrucibleTrain.Ports.HubClient`:

```elixir
defmodule CrucibleKitchen.Adapters.HfHub.HubClient do
  @behaviour CrucibleTrain.Ports.HubClient

  @impl true
  def download(adapter_opts, opts) do
    HfHub.Download.hf_hub_download(opts)
  end

  @impl true
  def snapshot(adapter_opts, opts) do
    HfHub.Download.snapshot_download(opts)
  end
end
```

#### Noop Adapters

Each port has a corresponding noop adapter for testing:

- `CrucibleKitchen.Adapters.Noop.TrainingClient`
- `CrucibleKitchen.Adapters.Noop.DatasetStore`
- `CrucibleKitchen.Adapters.Noop.HubClient`
- `CrucibleKitchen.Adapters.Noop.BlobStore`

---

## Workflow Engine

### Workflow Definition

CrucibleKitchen workflows are declarative stage compositions:

```elixir
defmodule CrucibleKitchen.Workflows.Supervised do
  use CrucibleKitchen.Workflow

  workflow do
    stage :load_dataset, Stages.LoadDataset
    stage :init_session, Stages.InitSession
    stage :init_tokenizer, Stages.InitTokenizer
    stage :build_dataset, Stages.BuildSupervisedDataset

    loop :epochs, over: fn ctx -> 0..(ctx.config.num_epochs - 1) end do
      stage :set_epoch, Stages.SetEpoch

      loop :batches, over: fn ctx -> ctx.state.dataset end do
        stage :get_batch, Stages.GetBatch
        stage :forward_backward, Stages.ForwardBackward
        stage :await_fb, Stages.AwaitFuture, key: :fb_future
        stage :optim_step, Stages.OptimStep
        stage :await_optim, Stages.AwaitFuture, key: :optim_future
        stage :log_step_metrics, Stages.LogStepMetrics
      end

      conditional fn ctx -> should_checkpoint?(ctx) end do
        stage :checkpoint, Stages.SaveCheckpoint
      end
    end

    stage :save_final, Stages.SaveFinalWeights
  end
end
```

### Stage Behaviour

Stages implement the `CrucibleKitchen.Stage` behaviour:

```elixir
defmodule CrucibleKitchen.Stages.ForwardBackward do
  use CrucibleKitchen.Stage

  @impl true
  def name, do: :forward_backward

  @impl true
  def execute(%{adapters: adapters, state: state} = context) do
    training_client = adapters.training_client
    batch = state.current_batch

    with {:ok, future} <- training_client.forward_backward(state.session, batch),
         {:ok, result} <- training_client.await(future) do
      {:ok, Context.put_state(context, :last_fb_result, result)}
    end
  end

  @impl true
  def validate(context) do
    if context.state[:session], do: :ok, else: {:error, :no_session}
  end
end
```

### Context Flow

A context map flows through the workflow, accumulating state:

```elixir
%CrucibleKitchen.Context{
  recipe: :sl_basic_v2,
  workflow: CrucibleKitchen.Workflows.Supervised,
  config: %{model: "...", epochs: 3, ...},
  adapters: %{
    training_client: TinkexAdapter,
    dataset_store: HfDatasetsAdapter
  },
  state: %{
    session: <session>,
    dataset: <dataset>,
    current_epoch: 0,
    current_step: 0,
    metrics: []
  },
  started_at: ~U[2025-12-27 10:00:00Z],
  current_stage: :forward_backward
}
```

---

## Data Flow

### Training Data Pipeline

```
+-------------------+
|  HuggingFace Hub  |
|  (no_robots)      |
+-------------------+
         |
         v
+-------------------+
|  HfDatasetsEx     |  <-- DatasetStore adapter
|  load_dataset()   |
+-------------------+
         |
         v
+-------------------+
|  NoRobots Module  |
|  sample_to_msgs() |
+-------------------+
         |
         v
+-------------------+
|  CrucibleTrain    |
|  Renderer         |  <-- Llama3, Qwen3, etc.
|  (tokenization)   |
+-------------------+
         |
         v
+-------------------+
|  CrucibleTrain    |
|  Types.Datum      |  <-- model_input + weights
+-------------------+
         |
         v
+-------------------+
|  Tinkex Adapter   |  <-- TrainingClient adapter
|  forward_backward |
+-------------------+
         |
         v
+-------------------+
|  Tinker Platform  |
|  (GPU training)   |
+-------------------+
```

### Datum Structure

Training data is represented as `CrucibleTrain.Types.Datum`:

```elixir
%CrucibleTrain.Types.Datum{
  model_input: %CrucibleTrain.Types.ModelInput{
    chunks: [
      %CrucibleTrain.Types.EncodedTextChunk{
        tokens: [128000, 128006, 882, 128007, ...],
        type: "encoded_text"
      }
    ]
  },
  loss_fn_inputs: %{
    targets: %CrucibleTrain.Types.TensorData{
      data: <<...>>,
      dtype: :s32,
      shape: [sequence_length]
    },
    weights: %CrucibleTrain.Types.TensorData{
      data: <<...>>,
      dtype: :f32,
      shape: [sequence_length]
    }
  }
}
```

---

## HuggingFace Integration

### HfDatasetsEx

Provides dataset loading from HuggingFace Hub:

```elixir
# Load a dataset
{:ok, dataset} = HfDatasetsEx.load_dataset("HuggingFaceH4/no_robots", split: "train")

# Operations
dataset = HfDatasetsEx.Dataset.shuffle(dataset, seed: 42)
dataset = HfDatasetsEx.Dataset.take(dataset, 1000)
samples = HfDatasetsEx.Dataset.to_list(dataset)
```

### HfHub

Provides model and file downloads:

```elixir
# Download a single file
{:ok, path} = HfHub.Download.hf_hub_download(
  repo_id: "meta-llama/Llama-3.1-8B",
  filename: "config.json"
)

# Download entire model
{:ok, path} = HfHub.Download.snapshot_download(
  repo_id: "meta-llama/Llama-3.1-8B"
)
```

### NoRobots Dataset Module

TinkexCookbook provides a high-level wrapper for the NoRobots dataset:

```elixir
# Load samples
{:ok, samples} = TinkexCookbook.Datasets.NoRobots.load(
  split: "train",
  limit: 100,
  shuffle_seed: 42
)

# Create supervised dataset with lazy rendering
dataset = TinkexCookbook.Datasets.NoRobots.create_supervised_dataset(samples,
  renderer_module: CrucibleTrain.Renderers.Llama3,
  renderer_state: state,
  train_on_what: :all_assistant_messages,
  batch_size: 32,
  max_length: 32768
)
```

---

## Tinkex SDK Integration

### Service Client

The Tinkex SDK provides gRPC communication with the Tinker platform:

```elixir
# Create config
tinkex_config = Tinkex.Config.new(
  api_key: System.get_env("TINKER_API_KEY"),
  base_url: "https://tinker.thinkingmachines.dev/services/tinker-prod"
)

# Start service client
{:ok, service_client} = Tinkex.ServiceClient.start_link(config: tinkex_config)

# Create training client with LoRA
lora_config = %Tinkex.Types.LoraConfig{rank: 32}
{:ok, training_client} = Tinkex.ServiceClient.create_lora_training_client(
  service_client,
  "meta-llama/Llama-3.1-8B",
  lora_config: lora_config
)
```

### Training Operations

```elixir
# Get tokenizer
{:ok, tokenizer} = Tinkex.TrainingClient.get_tokenizer(training_client)

# Forward-backward (async)
{:ok, fb_task} = Tinkex.TrainingClient.forward_backward(
  training_client,
  datums,
  :cross_entropy
)
{:ok, fb_result} = Task.await(fb_task, :infinity)

# Optimizer step (async)
adam_params = %Tinkex.Types.AdamParams{
  learning_rate: 2.0e-4,
  beta1: 0.9,
  beta2: 0.999,
  eps: 1.0e-8
}
{:ok, optim_task} = Tinkex.TrainingClient.optim_step(training_client, adam_params)
{:ok, optim_result} = Task.await(optim_task, :infinity)

# Checkpointing
{:ok, save_task} = Tinkex.TrainingClient.save_state(training_client, "checkpoint_001")
{:ok, _} = Task.await(save_task, :infinity)
```

---

## Snakepit Python Bridge

Snakepit provides Python interop for libraries without Elixir equivalents:

### Use Cases

- `sympy` - Symbolic mathematics verification
- `pylatexenc` - LaTeX parsing and rendering
- `math_verify` - Mathematical expression verification

### Architecture

```
+------------------+      gRPC      +------------------+
|   Elixir App     | <-----------> |  Python Worker   |
|   (Snakepit)     |               |  (snakepit_py)   |
+------------------+               +------------------+
```

### Usage Pattern

```elixir
# Start worker pool
{:ok, pool} = Snakepit.Pool.start_link(
  workers: 4,
  python_path: "/path/to/venv/bin/python"
)

# Call Python function
{:ok, result} = Snakepit.call(pool, "sympy", "simplify", ["x**2 + 2*x + 1"])
```

---

## Creating New Recipes

### Step 1: Define the Recipe Module

```elixir
defmodule TinkexCookbook.Recipes.MyCustomRecipe do
  use CrucibleKitchen.Recipe

  @impl true
  def name, do: :my_custom_recipe

  @impl true
  def description, do: "My custom training recipe"

  @impl true
  def default_config do
    %{
      model: "meta-llama/Llama-3.1-8B",
      epochs: 1,
      batch_size: 64,
      learning_rate: 1.0e-4,
      # Custom options
      my_option: true
    }
  end

  @impl true
  def required_adapters do
    [:training_client, :dataset_store]
  end

  @impl true
  def optional_adapters do
    [:blob_store, :metrics_store]
  end

  @impl true
  def workflow do
    # Use existing or define custom workflow
    CrucibleKitchen.Workflows.Supervised.__workflow__()
  end

  @impl true
  def validate_config(config) do
    cond do
      config[:epochs] < 1 -> {:error, "epochs must be >= 1"}
      config[:batch_size] < 1 -> {:error, "batch_size must be >= 1"}
      true -> :ok
    end
  end

  # Convenience function
  def run(config, opts \\ []) do
    adapters = build_adapters(opts)
    CrucibleKitchen.run(__MODULE__, config, adapters: adapters)
  end

  defp build_adapters(opts) do
    %{
      training_client: {CrucibleKitchen.Adapters.Tinkex.TrainingClient, opts},
      dataset_store: {CrucibleKitchen.Adapters.HfDatasets.DatasetStore, []}
    }
  end
end
```

### Step 2: Add Custom Dataset (Optional)

```elixir
defmodule TinkexCookbook.Datasets.MyDataset do
  def load(opts \\ []) do
    HfDatasetsEx.load_dataset("my-org/my-dataset", opts)
  end

  def sample_to_messages(%{"prompt" => prompt, "response" => response}) do
    [
      CrucibleTrain.Renderers.Types.message("user", prompt),
      CrucibleTrain.Renderers.Types.message("assistant", response)
    ]
  end

  def create_supervised_dataset(samples, opts) do
    # Similar to NoRobots implementation
  end
end
```

### Step 3: Add Mix Task (Optional)

```elixir
defmodule Mix.Tasks.MyRecipe do
  use Mix.Task

  @shortdoc "Run my custom recipe"

  def run(args) do
    Mix.Task.run("app.start")
    config = parse_args(args)
    TinkexCookbook.Recipes.MyCustomRecipe.run(config)
  end

  defp parse_args(args) do
    # Parse key=value pairs
  end
end
```

---

## Configuration Patterns

### Environment Variables

```elixir
defmodule TinkexCookbook.Config do
  def tinker_api_key do
    System.get_env("TINKER_API_KEY") ||
      raise "TINKER_API_KEY environment variable not set"
  end

  def tinker_base_url do
    System.get_env("TINKER_BASE_URL") ||
      "https://tinker.thinkingmachines.dev/services/tinker-prod"
  end

  def hf_token do
    System.get_env("HUGGING_FACE_HUB_TOKEN")
  end
end
```

### Config Merging

Recipes merge defaults with user-provided config:

```elixir
def run(user_config, opts \\ []) do
  config = Map.merge(default_config(), normalize_config(user_config))
  # ...
end

defp normalize_config(config) when is_struct(config), do: Map.from_struct(config)
defp normalize_config(config) when is_map(config), do: config
```

### Adapter Options

Adapters accept options at runtime:

```elixir
# Override API key
adapters = SlBasicV2.default_adapters(
  api_key: "custom_key",
  base_url: "https://custom.endpoint.com"
)

# The adapter receives these in start_session/2
def start_session(adapter_opts, config) do
  api_key = Keyword.get(adapter_opts, :api_key) || System.get_env("TINKER_API_KEY")
  # ...
end
```

---

## Telemetry Events

CrucibleKitchen emits comprehensive telemetry:

```elixir
# Workflow lifecycle
[:crucible_kitchen, :workflow, :start]
[:crucible_kitchen, :workflow, :stop]
[:crucible_kitchen, :workflow, :exception]

# Stage lifecycle
[:crucible_kitchen, :stage, :start]
[:crucible_kitchen, :stage, :stop]
[:crucible_kitchen, :stage, :exception]

# Training metrics
[:crucible_kitchen, :training, :step]
[:crucible_kitchen, :training, :epoch]
[:crucible_kitchen, :training, :checkpoint]
```

### Attaching Handlers

```elixir
:telemetry.attach(
  "my-handler",
  [:crucible_kitchen, :training, :step],
  fn _event, measurements, metadata, _config ->
    IO.puts("Step #{metadata.step}: loss=#{measurements.loss}")
  end,
  nil
)
```

---

## Dependency Graph

```
tinkex_cookbook
    |
    +-- crucible_kitchen (orchestration)
    |       |
    |       +-- crucible_train (types, renderers)
    |       +-- crucible_ir (experiment specs)
    |       +-- crucible_framework (pipeline runner)
    |       +-- telemetry
    |
    +-- tinkex (Tinker SDK)
    |       |
    |       +-- grpc
    |       +-- tokenizers
    |
    +-- hf_datasets_ex (HuggingFace datasets)
    |       |
    |       +-- finch (HTTP)
    |
    +-- hf_hub (HuggingFace Hub)
    |
    +-- snakepit (Python bridge)
    |       |
    |       +-- grpc
    |
    +-- chz_ex (config/CLI parsing)
    +-- nx (tensor operations)
```

---

## Testing

### Unit Tests

Use noop adapters for isolated testing:

```elixir
defmodule TinkexCookbook.Recipes.SlBasicV2Test do
  use ExUnit.Case

  test "validate_config returns error for missing model" do
    assert {:error, _} = SlBasicV2.validate_config(%{model: nil})
  end

  test "default_config has required fields" do
    config = SlBasicV2.default_config()
    assert config[:model]
    assert config[:epochs]
  end
end
```

### Integration Tests

Use Mox for adapter mocking:

```elixir
Mox.defmock(CrucibleTrain.Ports.TrainingClientMock, for: CrucibleTrain.Ports.TrainingClient)

CrucibleTrain.Ports.TrainingClientMock
|> Mox.expect(:start_session, fn _opts, _config -> {:ok, :mock_session} end)
|> Mox.expect(:forward_backward, fn _opts, _session, _datums ->
  Task.async(fn -> {:ok, %{}} end)
end)
```

---

## Further Reading

- [CrucibleKitchen Architecture Design](https://github.com/North-Shore-AI/crucible_kitchen/docs/)
- [CrucibleTrain Documentation](https://github.com/North-Shore-AI/crucible_train/)
- [Tinkex SDK Documentation](https://github.com/North-Shore-AI/tinkex/)
- [Snakepit Python Bridge](https://github.com/North-Shore-AI/snakepit/)
