defmodule Mix.Tasks.SlBasic do
  @shortdoc "Run the sl_basic supervised learning recipe"

  @moduledoc """
  Runs the sl_basic supervised learning recipe.

  This is a convenience wrapper around `TinkexCookbook.Recipes.SlBasic.main/1`.

  ## Usage

      mix sl_basic [OPTIONS]

  ## Options

  All options use `key=value` syntax:

      mix sl_basic log_path=/tmp/my_run learning_rate=0.0002

  Available options:

    * `log_path` - Output directory for logs and checkpoints (default: `/tmp/tinkex-examples/sl_basic`)
    * `model_name` - Model to fine-tune (default: `meta-llama/Llama-3.1-8B`)
    * `learning_rate` - Learning rate (default: `0.0002`)
    * `lr_schedule` - LR schedule: `linear` or `constant` (default: `linear`)
    * `num_epochs` - Number of training epochs (default: `1`)
    * `batch_size` - Batch size (default: `128`)
    * `max_length` - Maximum sequence length (default: `32768`)
    * `lora_rank` - LoRA rank (default: `32`)
    * `eval_every` - Run evaluation every N steps (default: `8`)
    * `save_every` - Save checkpoint every N steps (default: `20`)
    * `n_train_samples` - Limit training samples (default: all)
    * `train_on_what` - Weight strategy: `all_assistant_messages`, `last_assistant_message`, etc.
    * `behavior_if_exists` - What to do if log_path exists: `delete`, `resume`, `ask`, `raise`

  ## Environment Variables

    * `TINKER_API_KEY` - Required. Your Tinker API key.
    * `TINKER_BASE_URL` - Optional. Tinker API base URL.

  ## Examples

      # Basic run with defaults
      mix sl_basic

      # Custom output directory and learning rate
      mix sl_basic log_path=/tmp/experiment1 learning_rate=0.0001

      # Limit samples for quick testing
      mix sl_basic n_train_samples=100 num_epochs=1

      # Use a different model
      mix sl_basic model_name=meta-llama/Llama-3.2-3B batch_size=64
  """

  use Mix.Task

  alias TinkexCookbook.Recipes.SlBasic

  @impl Mix.Task
  def run(args) do
    # Ensure the application is started (loads configs, starts dependencies)
    Mix.Task.run("app.start")

    # Delegate to the recipe's main function
    case SlBasic.main(args) do
      :ok ->
        Mix.shell().info("Training completed successfully!")

      {:error, {:missing_env, var}} ->
        Mix.shell().error("Missing environment variable: #{var}")
        Mix.shell().error("Set it with: export #{var}=your_value")
        exit({:shutdown, 1})

      {:error, {:tokenizer_failed, %{message: msg}}} ->
        Mix.shell().error("Tokenizer download failed: #{msg}")
        Mix.shell().error("")
        Mix.shell().error("This usually means HuggingFace authentication is required.")
        Mix.shell().error("Set your token: export HUGGING_FACE_HUB_TOKEN=hf_your_token")
        exit({:shutdown, 1})

      {:error, {:training_client_failed, %{message: msg, data: data}}} ->
        Mix.shell().error("Training client failed: #{msg}")
        if data, do: Mix.shell().error("Details: #{inspect(data)}")
        exit({:shutdown, 1})

      {:error, {:dataset_load_failed, reason}} ->
        Mix.shell().error("Dataset loading failed: #{inspect(reason)}")
        exit({:shutdown, 1})

      {:error, reason} ->
        Mix.shell().error("Training failed: #{inspect(reason)}")
        exit({:shutdown, 1})
    end
  end
end
