defmodule Mix.Tasks.Kitchen.Run do
  @shortdoc "Run a CrucibleKitchen workflow"
  @moduledoc """
  Run a CrucibleKitchen workflow via CLI.

  ## Usage

      mix kitchen.run WORKFLOW [options]

  ## Available Workflows

  - `:supervised` - Supervised fine-tuning (SFT)
  - `:sl_basic_v2` - Supervised learning with Tinkex backend

  ## Options

  - `--model` / `-m` - Model name (default: meta-llama/Llama-3.1-8B)
  - `--dataset` / `-d` - Dataset name (default: HuggingFaceH4/no_robots)
  - `--epochs` / `-e` - Number of epochs (default: 1)
  - `--batch-size` / `-b` - Batch size (default: 128)
  - `--learning-rate` / `-l` - Learning rate (default: 2.0e-4)
  - `--lora-rank` / `-r` - LoRA rank (default: 32)
  - `--save-every` - Checkpoint every N steps (default: 20)
  - `--eval-every` - Evaluate every N steps (default: 0)
  - `--dry-run` - Validate configuration without executing

  ## Environment Variables

  - `TINKER_API_KEY` - Required for Tinkex backend
  - `TINKER_BASE_URL` - Optional custom API endpoint

  ## Examples

      # Basic supervised training
      mix kitchen.run :supervised \\
        --model meta-llama/Llama-3.1-8B \\
        --dataset HuggingFaceH4/no_robots \\
        --epochs 1

      # Using sl_basic_v2 recipe
      mix kitchen.run :sl_basic_v2 \\
        --model meta-llama/Llama-3.1-8B \\
        --epochs 2 \\
        --batch-size 64

      # Dry run to validate config
      mix kitchen.run :supervised --model my-model --dry-run

  ## Pipeline

  The supervised workflow executes:
  1. Load dataset from HuggingFace
  2. Initialize training session with Tinkex
  3. Training loop (forward/backward, optimizer step)
  4. Final evaluation (accuracy, F1, precision, recall)
  5. Model registration with lineage tracking
  6. Cleanup

  ## Results

  On success, the task outputs:
  - Training metrics (loss, learning rate)
  - Evaluation metrics (accuracy, F1)
  - Registered model ID and version
  """

  use Mix.Task

  require Logger

  alias CrucibleKitchen.Adapters.HfDatasets.DatasetStore, as: HfDatasetsAdapter
  alias CrucibleKitchen.Adapters.Noop.Evaluator, as: NoopEvaluator
  alias CrucibleKitchen.Adapters.Noop.ModelRegistry, as: NoopModelRegistry
  alias CrucibleKitchen.Adapters.Tinkex.TrainingClient, as: TinkexAdapter

  @switches [
    model: :string,
    dataset: :string,
    epochs: :integer,
    batch_size: :integer,
    learning_rate: :float,
    lora_rank: :integer,
    save_every: :integer,
    eval_every: :integer,
    max_length: :integer,
    dry_run: :boolean,
    help: :boolean
  ]

  @aliases [
    m: :model,
    d: :dataset,
    e: :epochs,
    b: :batch_size,
    l: :learning_rate,
    r: :lora_rank,
    h: :help
  ]

  @impl true
  def run(args) do
    Application.ensure_all_started(:tinkex_cookbook)

    {opts, rest, _} = OptionParser.parse(args, switches: @switches, aliases: @aliases)

    if opts[:help] do
      print_help()
    else
      workflow = parse_workflow(rest)
      run_workflow(workflow, opts)
    end
  end

  defp parse_workflow([]), do: :supervised

  defp parse_workflow([workflow | _]) when is_binary(workflow) do
    workflow
    |> String.trim_leading(":")
    |> String.to_atom()
  end

  defp run_workflow(workflow, opts) do
    config = build_config(opts)
    adapters = build_adapters()
    dry_run = Keyword.get(opts, :dry_run, false)

    print_banner(workflow, config)

    result = CrucibleKitchen.run(workflow, config, adapters: adapters, dry_run: dry_run)

    case result do
      {:ok, result} when dry_run ->
        Mix.shell().info("\n✅ Configuration validated successfully!")
        Mix.shell().info("   Duration: #{result.duration_ms}ms")

      {:ok, result} ->
        print_success(result)

      {:error, {:missing_adapters, missing}} ->
        Mix.shell().error("\n❌ Missing required adapters: #{inspect(missing)}")
        System.halt(1)

      {:error, reason} ->
        Mix.shell().error("\n❌ Workflow failed: #{inspect(reason)}")
        System.halt(1)
    end
  end

  defp build_config(opts) do
    %{
      model: Keyword.get(opts, :model, "meta-llama/Llama-3.1-8B"),
      dataset: Keyword.get(opts, :dataset, "HuggingFaceH4/no_robots"),
      epochs: Keyword.get(opts, :epochs, 1),
      batch_size: Keyword.get(opts, :batch_size, 128),
      learning_rate: Keyword.get(opts, :learning_rate, 2.0e-4),
      lora_rank: Keyword.get(opts, :lora_rank, 32),
      save_every: Keyword.get(opts, :save_every, 20),
      eval_every: Keyword.get(opts, :eval_every, 0),
      max_length: Keyword.get(opts, :max_length, 32_768),
      lr_schedule: :linear
    }
  end

  defp build_adapters do
    %{
      training_client: {TinkexAdapter, []},
      dataset_store: {HfDatasetsAdapter, []},
      evaluator: {NoopEvaluator, []},
      model_registry: {NoopModelRegistry, []}
    }
  end

  defp print_banner(workflow, config) do
    Mix.shell().info("""

    ╔══════════════════════════════════════════════════════════╗
    ║              🍳 CrucibleKitchen Training                 ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Workflow: #{String.pad_trailing(inspect(workflow), 43)}║
    ║  Model:    #{String.pad_trailing(config.model, 43)}║
    ║  Dataset:  #{String.pad_trailing(config.dataset, 43)}║
    ║  Epochs:   #{String.pad_trailing(Integer.to_string(config.epochs), 43)}║
    ╚══════════════════════════════════════════════════════════╝
    """)
  end

  defp print_success(result) do
    Mix.shell().info("\n✅ Workflow completed successfully!")
    Mix.shell().info("   Duration: #{result.duration_ms}ms")

    state = result.context.state

    if global_step = state[:global_step] do
      Mix.shell().info("   Total steps: #{global_step}")
    end

    if eval_results = state[:eval_results] do
      Mix.shell().info("\n📊 Evaluation Results:")

      for {metric, value} <- eval_results,
          metric not in [:sample_count, :step, :evaluated, :skipped, :note] do
        Mix.shell().info("   #{metric}: #{format_metric(value)}")
      end
    end

    if model = state[:registered_model] do
      Mix.shell().info("\n📦 Registered Model:")
      Mix.shell().info("   Name: #{model.name}")
      Mix.shell().info("   Version: #{model.version}")
      Mix.shell().info("   ID: #{model.id}")
    end
  end

  defp format_metric(value) when is_float(value), do: Float.round(value, 4)
  defp format_metric(value), do: inspect(value)

  defp print_help do
    Mix.shell().info(@moduledoc)
  end
end
