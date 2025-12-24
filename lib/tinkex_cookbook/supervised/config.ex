defmodule TinkexCookbook.Supervised.ChatDatasetBuilderCommonConfig do
  @moduledoc """
  Common configuration shared by all chat dataset builders.

  This struct mirrors Python's `ChatDatasetBuilderCommonConfig` from
  `tinker_cookbook/supervised/types.py`.

  ## Fields

  - `model_name_for_tokenizer` - Model name used to load the tokenizer
  - `renderer_name` - Name of the renderer to use (e.g., "llama3", "qwen3")
  - `max_length` - Maximum sequence length for truncation
  - `batch_size` - Number of examples per batch
  - `train_on_what` - Optional training target strategy (TrainOnWhat value)
  """

  @type t :: %__MODULE__{
          model_name_for_tokenizer: String.t(),
          renderer_name: String.t(),
          max_length: pos_integer() | nil,
          batch_size: pos_integer(),
          train_on_what: String.t() | nil
        }

  @enforce_keys [:model_name_for_tokenizer, :renderer_name, :batch_size]
  defstruct [
    :model_name_for_tokenizer,
    :renderer_name,
    :max_length,
    :batch_size,
    :train_on_what
  ]

  @doc """
  Creates a new common config with the given options.
  """
  @spec new(keyword()) :: t()
  def new(opts) when is_list(opts) do
    struct!(__MODULE__, opts)
  end
end

defmodule TinkexCookbook.Supervised.Config do
  @moduledoc """
  Configuration for supervised fine-tuning.

  This struct mirrors Python's `train.Config` from
  `tinker_cookbook/supervised/train.py`.

  ## Required Fields

  - `log_path` - Directory path for logs and checkpoints
  - `model_name` - Base model name for training
  - `dataset_builder` - Tuple of `{builder_type, common_config}`

  ## Training Parameters

  - `learning_rate` - Learning rate (default: 1e-4)
  - `lr_schedule` - LR schedule type: "linear", "cosine", "constant" (default: "linear")
  - `num_epochs` - Number of training epochs (default: 1)

  ## Model Parameters

  - `lora_rank` - LoRA adapter rank (default: 32)
  - `load_checkpoint_path` - Optional path to resume from

  ## Checkpointing and Evaluation

  - `save_every` - Save checkpoint every N steps, 0 to disable (default: 20)
  - `eval_every` - Run evaluation every N steps, 0 to disable (default: 10)

  ## Adam Optimizer Parameters

  - `adam_beta1` - Adam beta1 (default: 0.9)
  - `adam_beta2` - Adam beta2 (default: 0.95)
  - `adam_eps` - Adam epsilon (default: 1e-8)
  """

  alias TinkexCookbook.Supervised.ChatDatasetBuilderCommonConfig

  @type dataset_builder ::
          {:no_robots, ChatDatasetBuilderCommonConfig.t()}
          | {:tulu3, ChatDatasetBuilderCommonConfig.t()}
          | {:from_file, ChatDatasetBuilderCommonConfig.t(), String.t()}

  @type lr_schedule :: String.t()

  @type t :: %__MODULE__{
          log_path: String.t(),
          model_name: String.t(),
          load_checkpoint_path: String.t() | nil,
          dataset_builder: dataset_builder(),
          learning_rate: float(),
          lr_schedule: lr_schedule(),
          num_epochs: pos_integer(),
          lora_rank: pos_integer(),
          base_url: String.t() | nil,
          save_every: non_neg_integer(),
          eval_every: non_neg_integer(),
          adam_beta1: float(),
          adam_beta2: float(),
          adam_eps: float(),
          wandb_project: String.t() | nil,
          wandb_name: String.t() | nil
        }

  @enforce_keys [:log_path, :model_name, :dataset_builder]
  defstruct [
    :log_path,
    :model_name,
    :load_checkpoint_path,
    :dataset_builder,
    :base_url,
    :wandb_project,
    :wandb_name,
    learning_rate: 1.0e-4,
    lr_schedule: "linear",
    num_epochs: 1,
    lora_rank: 32,
    save_every: 20,
    eval_every: 10,
    adam_beta1: 0.9,
    adam_beta2: 0.95,
    adam_eps: 1.0e-8
  ]

  @doc """
  Creates a new config with the given options.
  """
  @spec new(keyword()) :: t()
  def new(opts) when is_list(opts) do
    struct!(__MODULE__, opts)
  end

  @doc """
  Expands the log_path, handling ~ for home directory.
  """
  @spec expand_log_path(t()) :: t()
  def expand_log_path(%__MODULE__{log_path: path} = config) do
    %{config | log_path: Path.expand(path)}
  end
end
