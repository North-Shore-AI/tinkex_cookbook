# credo:disable-for-this-file Credo.Check.Refactor.Nesting
defmodule TinkexCookbook.Distillation.TeacherConfig do
  @moduledoc """
  Configuration for a teacher model.
  """

  use ChzEx.Schema

  chz_schema do
    field(:base_model, :string)
    field(:load_checkpoint_path, :string, default: nil)
  end
end

defmodule TinkexCookbook.Distillation.DistillationDatasetConfig do
  @moduledoc """
  Configuration for a distillation dataset.
  """

  use ChzEx.Schema

  chz_schema do
    field(:dataset_builder, :any, virtual: true)
    field(:teacher_config, :any, virtual: true)
    field(:groups_per_batch, :integer)
  end
end

defmodule TinkexCookbook.Distillation.CompositeDataset do
  @moduledoc """
  Wraps multiple RL datasets and samples from each.
  """

  @type t :: %__MODULE__{
          datasets: [struct()],
          groups_per_batch_list: [pos_integer()]
        }

  defstruct [:datasets, :groups_per_batch_list]

  @spec new([struct()], [pos_integer()]) :: t()
  def new(datasets, groups_per_batch_list) do
    %__MODULE__{datasets: datasets, groups_per_batch_list: groups_per_batch_list}
  end

  @spec length(t()) :: non_neg_integer()
  def length(%__MODULE__{datasets: []}), do: 0

  def length(%__MODULE__{datasets: datasets}) do
    datasets
    |> Enum.map(& &1.__struct__.length(&1))
    |> Enum.min()
  end

  @spec get_batch(t(), non_neg_integer()) :: {[struct()], [non_neg_integer()]}
  def get_batch(%__MODULE__{} = dataset, i_batch) do
    {builders, indices} =
      dataset.datasets
      |> Enum.with_index()
      |> Enum.zip(dataset.groups_per_batch_list)
      |> Enum.reduce({[], []}, fn {{ds, idx}, groups_per_batch}, {all_builders, all_indices} ->
        env_group_builders = ds.__struct__.get_batch(ds, i_batch)

        if Kernel.length(env_group_builders) != groups_per_batch do
          raise ArgumentError,
                "Dataset #{idx} returned #{Kernel.length(env_group_builders)} items, expected #{groups_per_batch}"
        end

        {
          all_builders ++ env_group_builders,
          all_indices ++ List.duplicate(idx, groups_per_batch)
        }
      end)

    {builders, indices}
  end
end

defmodule TinkexCookbook.Distillation.PromptOnlyEnv do
  @moduledoc """
  Environment that only provides prompts with no rewards.
  """

  use TinkexCookbook.RL.ProblemEnv

  defstruct [:prompt, :renderer_module, :renderer_state, convo_prefix: nil, format_coef: 0.0]

  @type t :: %__MODULE__{
          prompt: String.t(),
          renderer_module: module(),
          renderer_state: map(),
          convo_prefix: [map()] | nil,
          format_coef: float()
        }

  @impl true
  def get_question(%__MODULE__{prompt: prompt}), do: prompt

  @impl true
  def check_format(_env, _sample_str), do: true

  @impl true
  def check_answer(_env, _sample_str), do: false

  @impl true
  def get_reference_answer(_env), do: ""
end

defmodule TinkexCookbook.Distillation.PromptOnlyDataset do
  @moduledoc """
  Dataset that yields prompt-only environments.
  """

  @behaviour TinkexCookbook.RL.RLDataset

  alias TinkexCookbook.Renderers.Helpers
  alias TinkexCookbook.RL.ProblemGroupBuilder

  defstruct [
    :prompts,
    :batch_size,
    :group_size,
    :renderer_module,
    :renderer_state,
    :tokenizer,
    :max_prompt_tokens,
    :convo_prefix,
    dataset_name: "prompts"
  ]

  @type t :: %__MODULE__{
          prompts: [String.t()],
          batch_size: pos_integer(),
          group_size: pos_integer(),
          renderer_module: module(),
          renderer_state: map(),
          tokenizer: term(),
          max_prompt_tokens: pos_integer() | nil,
          convo_prefix: [map()] | nil,
          dataset_name: String.t()
        }

  @impl true
  def get_batch(%__MODULE__{} = dataset, index) do
    batch_start = index * dataset.batch_size
    batch_end = min((index + 1) * dataset.batch_size, Kernel.length(dataset.prompts))

    if batch_start >= batch_end do
      raise ArgumentError, "Incorrect batch size"
    end

    dataset.prompts
    |> Enum.slice(batch_start, batch_end - batch_start)
    |> Enum.map(fn prompt ->
      truncated = truncate_prompt(dataset, prompt)

      %ProblemGroupBuilder{
        env_thunk: fn ->
          %TinkexCookbook.Distillation.PromptOnlyEnv{
            prompt: truncated,
            renderer_module: dataset.renderer_module,
            renderer_state: dataset.renderer_state,
            convo_prefix: dataset.convo_prefix,
            format_coef: 0.0
          }
        end,
        num_envs: dataset.group_size,
        dataset_name: dataset.dataset_name
      }
    end)
  end

  @impl true
  def length(%__MODULE__{} = dataset) do
    div(Kernel.length(dataset.prompts) + dataset.batch_size - 1, dataset.batch_size)
  end

  defp truncate_prompt(%__MODULE__{max_prompt_tokens: nil}, prompt), do: prompt

  defp truncate_prompt(%__MODULE__{max_prompt_tokens: max_tokens, tokenizer: tokenizer}, prompt) do
    tokens = Helpers.encode(tokenizer, prompt, add_special_tokens: false)

    if Kernel.length(tokens) > max_tokens do
      tokens
      |> Enum.take(max_tokens)
      |> then(&Helpers.decode(tokenizer, &1))
    else
      prompt
    end
  end
end

defmodule TinkexCookbook.Distillation.Datasets do
  @moduledoc """
  Dataset utilities for on-policy distillation.
  """

  require Logger

  @spec load_deepmath_prompts(String.t()) :: [String.t()] | nil
  def load_deepmath_prompts(split \\ "train") do
    case HfDatasetsEx.load_dataset("zwhe99/DeepMath-103K", split: split) do
      {:ok, dataset} ->
        dataset.items
        |> Enum.map(&Map.get(&1, "question"))
        |> Enum.filter(&is_binary/1)

      {:error, reason} ->
        Logger.warning("Could not load #{split} split for DeepMath: #{inspect(reason)}")
        nil
    end
  rescue
    exception ->
      Logger.warning(
        "Could not load #{split} split for DeepMath: #{Exception.message(exception)}"
      )

      nil
  end

  @spec load_tulu3_prompts() :: [String.t()] | nil
  def load_tulu3_prompts do
    case HfDatasetsEx.load_dataset("allenai/tulu-3-sft-mixture", split: "train") do
      {:ok, dataset} ->
        dataset.items
        |> Enum.reduce([], fn row, acc ->
          messages = Map.get(row, "messages", [])

          first_user =
            Enum.find_value(messages, fn msg ->
              if Map.get(msg, "role") == "user" do
                Map.get(msg, "content")
              else
                nil
              end
            end)

          if is_binary(first_user) do
            [first_user | acc]
          else
            acc
          end
        end)
        |> Enum.reverse()

      {:error, reason} ->
        Logger.warning("Could not load Tulu3 dataset: #{inspect(reason)}")
        nil
    end
  rescue
    exception ->
      Logger.warning("Could not load Tulu3 dataset: #{Exception.message(exception)}")
      nil
  end

  defmodule PromptOnlyDatasetBuilder do
    @moduledoc """
    Builder for prompt-only datasets.
    """

    use ChzEx.Schema

    alias TinkexCookbook.Distillation.{Datasets, PromptOnlyDataset}
    alias TinkexCookbook.Renderers
    alias TinkexCookbook.TokenizerUtils

    chz_schema do
      field(:dataset_name, :string)
      field(:groups_per_batch, :integer)
      field(:group_size, :integer)
      field(:model_name_for_tokenizer, :string)
      field(:renderer_name, :string)
      field(:convo_prefix, :any, default: nil, virtual: true)
      field(:max_prompt_tokens, :integer, default: 1024)
    end

    @spec build(struct()) :: {PromptOnlyDataset.t(), PromptOnlyDataset.t() | nil}
    def build(%__MODULE__{} = cfg) do
      {:ok, tokenizer} = TokenizerUtils.get_tokenizer(cfg.model_name_for_tokenizer)
      {:ok, renderer_module, extra_opts} = Renderers.lookup(cfg.renderer_name)
      {:ok, renderer_state} = renderer_module.init([{:tokenizer, tokenizer} | extra_opts])

      {train_prompts, test_prompts} =
        case cfg.dataset_name do
          "deepmath" ->
            {Datasets.load_deepmath_prompts("train"), Datasets.load_deepmath_prompts("test")}

          "tulu3" ->
            {Datasets.load_tulu3_prompts(), nil}

          other ->
            raise ArgumentError, "Unknown dataset: #{other}"
        end

      if train_prompts == nil do
        raise ArgumentError, "Could not load train split for #{cfg.dataset_name}"
      end

      train = %PromptOnlyDataset{
        prompts: train_prompts,
        batch_size: cfg.groups_per_batch,
        group_size: cfg.group_size,
        renderer_module: renderer_module,
        renderer_state: renderer_state,
        tokenizer: tokenizer,
        max_prompt_tokens: cfg.max_prompt_tokens,
        convo_prefix: cfg.convo_prefix,
        dataset_name: cfg.dataset_name
      }

      test =
        if test_prompts != nil do
          %PromptOnlyDataset{
            prompts: test_prompts,
            batch_size: cfg.groups_per_batch,
            group_size: 1,
            renderer_module: renderer_module,
            renderer_state: renderer_state,
            tokenizer: tokenizer,
            max_prompt_tokens: cfg.max_prompt_tokens,
            convo_prefix: cfg.convo_prefix,
            dataset_name: cfg.dataset_name <> "_test"
          }
        else
          nil
        end

      {train, test}
    end
  end
end
