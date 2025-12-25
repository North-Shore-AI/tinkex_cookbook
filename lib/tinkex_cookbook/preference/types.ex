defmodule TinkexCookbook.Preference.Comparison do
  @moduledoc """
  Comparison between two completions for a shared prompt conversation.
  """

  alias TinkexCookbook.Renderers.Types

  @type t :: %__MODULE__{
          prompt_conversation: [Types.Message.t() | map()],
          completion_a: [Types.Message.t() | map()],
          completion_b: [Types.Message.t() | map()]
        }

  defstruct [:prompt_conversation, :completion_a, :completion_b]

  @spec swap(t()) :: t()
  def swap(%__MODULE__{} = comparison) do
    %__MODULE__{
      comparison
      | completion_a: comparison.completion_b,
        completion_b: comparison.completion_a
    }
  end
end

defmodule TinkexCookbook.Preference.LabeledComparison do
  @moduledoc """
  Comparison with a label indicating the preferred completion.
  """

  alias TinkexCookbook.Preference.Comparison

  @type label :: String.t()

  @type t :: %__MODULE__{
          comparison: Comparison.t(),
          label: label()
        }

  defstruct [:comparison, :label]

  @spec swap(t()) :: t()
  def swap(%__MODULE__{} = labeled) do
    new_label =
      case labeled.label do
        "A" -> "B"
        "B" -> "A"
        "Tie" -> "Tie"
      end

    %__MODULE__{labeled | comparison: Comparison.swap(labeled.comparison), label: new_label}
  end
end

defmodule TinkexCookbook.Preference.ComparisonRenderer do
  @moduledoc """
  Behaviour for rendering comparisons into model inputs.
  """

  alias TinkexCookbook.Preference.{Comparison, LabeledComparison}
  alias TinkexCookbook.Types.ModelInput

  @callback build_generation_prompt(struct(), Comparison.t()) :: ModelInput.t()
  @callback to_model_input_weights(struct(), LabeledComparison.t()) :: {ModelInput.t(), [float()]}
  @callback tokenizer(struct()) :: term()

  @spec build_generation_prompt(struct(), Comparison.t()) :: ModelInput.t()
  def build_generation_prompt(%module{} = renderer, comparison) do
    module.build_generation_prompt(renderer, comparison)
  end

  @spec to_model_input_weights(struct(), LabeledComparison.t()) :: {ModelInput.t(), [float()]}
  def to_model_input_weights(%module{} = renderer, labeled) do
    module.to_model_input_weights(renderer, labeled)
  end

  @spec tokenizer(struct()) :: term()
  def tokenizer(%module{} = renderer) do
    module.tokenizer(renderer)
  end
end

defmodule TinkexCookbook.Preference.ComparisonRendererFromChatRenderer do
  @moduledoc """
  Comparison renderer that adapts an existing chat renderer.
  """

  alias TinkexCookbook.Preference.{Comparison, LabeledComparison}
  alias TinkexCookbook.Renderers.{Renderer, TrainOnWhat, Types}
  alias TinkexCookbook.Types.{EncodedTextChunk, ModelInput}

  defstruct [:renderer_module, :renderer_state]

  @type t :: %__MODULE__{
          renderer_module: module(),
          renderer_state: map()
        }

  @spec build_generation_prompt(t(), Comparison.t()) :: ModelInput.t()
  def build_generation_prompt(%__MODULE__{} = renderer, %Comparison{} = comparison) do
    convo = comparison_to_convo(comparison)

    {model_input, _state} =
      Renderer.build_generation_prompt(
        renderer.renderer_module,
        convo,
        "assistant",
        nil,
        renderer.renderer_state
      )

    model_input
  end

  @spec to_model_input_weights(t(), LabeledComparison.t()) :: {ModelInput.t(), [float()]}
  def to_model_input_weights(%__MODULE__{} = renderer, %LabeledComparison{} = labeled) do
    convo = comparison_to_convo(labeled.comparison)
    convo_with_pref = convo ++ [Types.message("assistant", labeled.label)]

    {model_input, weights} =
      Renderer.build_supervised_example(
        renderer.renderer_module,
        convo_with_pref,
        TrainOnWhat.last_assistant_message(),
        renderer.renderer_state
      )

    if Enum.any?(model_input.chunks, fn chunk -> not match?(%EncodedTextChunk{}, chunk) end) do
      raise ArgumentError, "Preference learning currently only supports text-only content."
    end

    tokens = ModelInput.all_tokens(model_input)

    first_weight_one_index =
      case Enum.find_index(weights, &(&1 == 1.0)) do
        nil -> raise ArgumentError, "No weight==1 token found for preference label"
        idx -> idx
      end

    truncated_tokens = Enum.take(tokens, first_weight_one_index + 1)
    truncated_weights = Enum.take(weights, first_weight_one_index + 1)

    {ModelInput.from_ints(truncated_tokens), truncated_weights}
  end

  @spec tokenizer(t()) :: term()
  def tokenizer(%__MODULE__{} = renderer) do
    Map.fetch!(renderer.renderer_state, :tokenizer)
  end

  defp comparison_to_convo(%Comparison{} = comparison) do
    [
      comparison.prompt_conversation,
      [Types.message("system", "==== Completion A ====")],
      comparison.completion_a,
      [Types.message("system", "==== Completion B ====")],
      comparison.completion_b,
      [Types.message("system", "==== Preference ====")]
    ]
    |> List.flatten()
  end
end

defmodule TinkexCookbook.Preference.PreferenceModel do
  @moduledoc """
  Behaviour for preference models that score comparisons.
  """

  alias TinkexCookbook.Preference.Comparison

  @callback score(struct(), Comparison.t()) :: float()

  @spec score(struct(), Comparison.t()) :: float()
  def score(%module{} = model, comparison) do
    module.score(model, comparison)
  end
end

defmodule TinkexCookbook.Preference.PreferenceModelBuilder do
  @moduledoc """
  Behaviour for building preference models.
  """

  @callback build(struct()) :: struct()

  @spec build(struct()) :: struct()
  def build(%module{} = builder) do
    module.build(builder)
  end
end

defmodule TinkexCookbook.Preference.PreferenceModelFromChatRenderer do
  @moduledoc """
  Preference model that uses a chat renderer + sampling client.
  """

  require Logger

  alias Tinkex.Types.SamplingParams
  alias TinkexCookbook.Preference.{Comparison, ComparisonRenderer}
  alias TinkexCookbook.Renderers.Helpers
  alias TinkexCookbook.Utils.TinkexConvert

  defstruct [:comparison_renderer, :sampling_client]

  @type t :: %__MODULE__{
          comparison_renderer: struct(),
          sampling_client: pid()
        }

  @spec score(t(), Comparison.t()) :: float()
  def score(%__MODULE__{} = model, %Comparison{} = comparison) do
    model_input =
      ComparisonRenderer.build_generation_prompt(model.comparison_renderer, comparison)

    tokenizer = ComparisonRenderer.tokenizer(model.comparison_renderer)
    sampling_params = %SamplingParams{temperature: 0.0, max_tokens: 1}

    with {:ok, task} <- sample_comparison(model, model_input, sampling_params),
         {:ok, response} <- Task.await(task, :infinity) do
      process_response(response, tokenizer)
    else
      {:error, reason} ->
        Logger.warning("Preference model sampling failed: #{inspect(reason)}")
        0.0
    end
  end

  defp sample_comparison(model, model_input, sampling_params) do
    model.sampling_client.__struct__.sample(
      model.sampling_client,
      TinkexConvert.model_input_to_tinkex(model_input),
      sampling_params,
      num_samples: 1
    )
  end

  defp process_response(response, tokenizer) do
    sequences = response.sequences || response[:sequences] || []

    case sequences do
      [sequence | _] ->
        tokens = sequence.tokens || sequence[:tokens] || []
        output = Helpers.decode(tokenizer, tokens) |> String.trim()
        parse_preference_output(output)

      [] ->
        Logger.warning("Preference model returned no sequences")
        0.0
    end
  end

  defp parse_preference_output(output) do
    case output do
      "A" ->
        -1.0

      "B" ->
        1.0

      "Tie" ->
        0.0

      _ ->
        Logger.warning("Invalid preference model output: #{inspect(output)}")
        0.0
    end
  end
end

defmodule TinkexCookbook.Preference.PreferenceModelBuilderFromChatRenderer do
  @moduledoc """
  Builder for preference models backed by chat renderers.
  """

  use ChzEx.Schema

  alias TinkexCookbook.Preference.{
    ComparisonRendererFromChatRenderer,
    PreferenceModelFromChatRenderer
  }

  alias TinkexCookbook.Renderers
  alias TinkexCookbook.TokenizerUtils

  @default_base_url "https://tinker.thinkingmachines.dev/services/tinker-prod"

  chz_schema do
    field(:renderer_name, :string)
    field(:model_name, :string)
    field(:rm_weights_path, :string)
    field(:base_url, :string, default: nil)
  end

  @spec build(%__MODULE__{}) :: PreferenceModelFromChatRenderer.t()
  def build(%__MODULE__{} = builder) do
    api_key = System.get_env("TINKER_API_KEY")

    if is_nil(api_key) do
      raise ArgumentError, "TINKER_API_KEY environment variable is required"
    end

    base_url = builder.base_url || System.get_env("TINKER_BASE_URL", @default_base_url)

    tinkex_config = Tinkex.Config.new(api_key: api_key, base_url: base_url)

    {:ok, service_client} = Tinkex.ServiceClient.start_link(config: tinkex_config)

    {:ok, tokenizer} = TokenizerUtils.get_tokenizer(builder.model_name)

    {:ok, renderer_module, extra_opts} = Renderers.lookup(builder.renderer_name)
    {:ok, renderer_state} = renderer_module.init([{:tokenizer, tokenizer} | extra_opts])

    {:ok, sampling_client} =
      Tinkex.ServiceClient.create_sampling_client(service_client,
        model_path: builder.rm_weights_path
      )

    %PreferenceModelFromChatRenderer{
      comparison_renderer: %ComparisonRendererFromChatRenderer{
        renderer_module: renderer_module,
        renderer_state: renderer_state
      },
      sampling_client: sampling_client
    }
  end
end
