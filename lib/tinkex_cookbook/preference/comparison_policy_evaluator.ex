defmodule TinkexCookbook.Preference.ComparisonPolicyEvaluator do
  @moduledoc """
  Evaluates a policy by comparing its completions to references using a reward model.

  This evaluator generates completions for each comparison prompt and scores
  them against reference completions using a preference model. It computes
  a win rate indicating how often the policy's completion is preferred.

  ## Reward Calculation

  For each comparison, the evaluator:
  1. Generates a new completion from the policy for the prompt
  2. Creates two comparisons: (reference, policy) and (policy, reference)
  3. Computes preference scores r_0 and r_1 for each ordering
  4. Normalizes: `(r_0 - r_1 + 2) / 4.0` → range [0, 1]

  The final win_rate is the mean across all comparisons.

  ## Example

      comparisons = [
        %Comparison{prompt_conversation: [...], completion_a: [...], completion_b: nil}
      ]

      evaluator = %ComparisonPolicyEvaluator{
        preference_model_builder: fn -> build_rm() end,
        comparisons: comparisons,
        renderer_module: Llama3,
        renderer_state: state,
        max_tokens: 1024
      }

      {:ok, metrics} = ComparisonPolicyEvaluator.evaluate(evaluator, sampling_client)
      # metrics = %{"win_rate" => 0.65, "stderr" => 0.05}

  See `tinker-cookbook/tinker_cookbook/preference/comparison_policy_evaluator.py` for Python reference.
  """

  require Logger

  alias TinkexCookbook.Completers.TinkexMessageCompleter
  alias TinkexCookbook.Eval.Evaluators.SamplingClientEvaluator
  alias TinkexCookbook.Preference.{Comparison, PreferenceModel}
  alias TinkexCookbook.Renderers.Types

  @behaviour SamplingClientEvaluator

  defstruct [
    :preference_model_builder,
    :comparisons,
    :renderer_module,
    :renderer_state,
    max_tokens: 1024,
    both_ways: true,
    content_preprocessor: nil
  ]

  @type t :: %__MODULE__{
          preference_model_builder: (-> struct()),
          comparisons: [Comparison.t()],
          renderer_module: module(),
          renderer_state: map(),
          max_tokens: pos_integer(),
          both_ways: boolean(),
          content_preprocessor: (String.t() -> String.t()) | nil
        }

  @doc """
  Evaluates the sampling client by comparing policy completions against references.

  Returns a map containing:
  - `"win_rate"`: Mean preference score (0-1, higher is better)
  - `"stderr"`: Standard error of the win rate

  ## Parameters

  - `evaluator`: The evaluator configuration
  - `sampling_client`: A Tinkex sampling client to generate completions

  ## Returns

  - `{:ok, %{"win_rate" => float(), "stderr" => float()}}` on success
  - `{:error, reason}` on failure
  """
  @impl true
  @spec evaluate(t(), pid()) :: {:ok, map()} | {:error, term()}
  def evaluate(%__MODULE__{} = evaluator, sampling_client) do
    preference_model = evaluator.preference_model_builder.()

    # Process all comparisons concurrently
    tasks =
      Enum.map(evaluator.comparisons, fn comparison ->
        Task.async(fn ->
          process_comparison(evaluator, sampling_client, preference_model, comparison)
        end)
      end)

    results = Task.await_many(tasks, :infinity)

    # Filter out errors and compute statistics
    scores = Enum.filter(results, &is_number/1)

    if scores != [] do
      win_rate = Enum.sum(scores) / length(scores)

      variance =
        Enum.reduce(scores, 0.0, fn s, acc -> acc + (s - win_rate) ** 2 end) / length(scores)

      stderr = :math.sqrt(variance) / :math.sqrt(length(scores))

      {:ok, %{"win_rate" => win_rate, "stderr" => stderr}}
    else
      {:error, :no_successful_comparisons}
    end
  end

  defp process_comparison(evaluator, sampling_client, preference_model, comparison) do
    case generate_completion(evaluator, sampling_client, comparison.prompt_conversation) do
      {:ok, new_completion_message} ->
        # Preprocess the content if needed
        content = Types.ensure_text(new_completion_message.content)

        preprocessed_content =
          if evaluator.content_preprocessor do
            evaluator.content_preprocessor.(content)
          else
            content
          end

        new_message = Types.message("assistant", preprocessed_content)

        # Create comparison with new completion as B
        new_comparison = %Comparison{
          comparison
          | completion_b: [new_message]
        }

        # Score both orderings
        r_0 = PreferenceModel.score(preference_model, new_comparison)
        r_1 = PreferenceModel.score(preference_model, Comparison.swap(new_comparison))

        # r_0, r_1 are in [-1, 1]
        # r_0 - r_1 is in [-2, 2]
        # Normalize to [0, 1]
        (r_0 - r_1 + 2) / 4.0

      {:error, reason} ->
        Logger.warning("Failed to generate completion: #{inspect(reason)}")
        nil
    end
  end

  defp generate_completion(evaluator, sampling_client, prompt_conversation) do
    # Use TinkexMessageCompleter for generation
    completer =
      TinkexMessageCompleter.new(
        sampling_client: sampling_client,
        renderer_module: evaluator.renderer_module,
        renderer_state: evaluator.renderer_state,
        max_tokens: evaluator.max_tokens
      )

    TinkexMessageCompleter.complete(completer, prompt_conversation)
  end
end

defmodule TinkexCookbook.Preference.ComparisonPolicyEvaluatorBuilder do
  @moduledoc """
  Builder for ComparisonPolicyEvaluator.

  Used with ChzEx for configuration serialization.
  """

  use ChzEx.Schema

  alias TinkexCookbook.Preference.ComparisonPolicyEvaluator
  alias TinkexCookbook.Renderers
  alias TinkexCookbook.TokenizerUtils

  chz_schema do
    field(:preference_model_builder, :any, virtual: true)
    field(:comparisons, {:array, :any}, virtual: true)
    field(:renderer_name, :string)
    field(:model_name_for_tokenizer, :string)
    field(:both_ways, :boolean, default: true)
    field(:max_tokens, :integer, default: 1024)
    field(:content_preprocessor, :any, default: nil, virtual: true)
  end

  @doc """
  Builds a ComparisonPolicyEvaluator from this builder.
  """
  @spec build(struct()) :: ComparisonPolicyEvaluator.t()
  def build(%__MODULE__{} = builder) do
    {:ok, tokenizer} = TokenizerUtils.get_tokenizer(builder.model_name_for_tokenizer)
    {:ok, renderer_module, extra_opts} = Renderers.lookup(builder.renderer_name)
    {:ok, renderer_state} = renderer_module.init([{:tokenizer, tokenizer} | extra_opts])

    %ComparisonPolicyEvaluator{
      preference_model_builder: builder.preference_model_builder,
      comparisons: builder.comparisons,
      renderer_module: renderer_module,
      renderer_state: renderer_state,
      max_tokens: builder.max_tokens,
      both_ways: builder.both_ways,
      content_preprocessor: builder.content_preprocessor
    }
  end
end
