defmodule TinkexCookbook.Eval.Runner do
  @moduledoc """
  Orchestrates evaluation using EvalEx and CrucibleHarness.

  The Runner provides a simple interface for running evaluations on
  fine-tuned models using the TinkexGenerate adapter.

  ## Usage

      # Setup
      sampling_client = Tinkex.SamplingClient.new(...)

      config = %{
        sampling_client: sampling_client,
        model: "meta-llama/Llama-3.1-8B",
        temperature: 0.7,
        max_tokens: 1024,
        stop: ["<|eot_id|>"]
      }

      # Load samples
      samples = [
        %{id: "1", input: "What is 2+2?", target: "4"},
        %{id: "2", input: "What is 3+3?", target: "6"}
      ]

      # Run evaluation
      {:ok, results} = Runner.run(samples, config)

      # Score results
      scored = Runner.score_results(results, :exact_match)

      # Compute metrics
      metrics = Runner.compute_metrics(scored)
      # => %{accuracy: 0.75, total: 4, correct: 3}

  ## Integration with EvalEx Tasks

      task = EvalEx.Task.new(
        id: "my_eval",
        name: "My Evaluation",
        dataset: samples,
        scorers: [EvalEx.Scorer.ExactMatch]
      )

      {:ok, results} = Runner.run_task(task, config)
  """

  require Logger

  alias TinkexCookbook.Eval.TinkexGenerate
  alias TinkexCookbook.Renderers.Types

  @type sample :: %{
          required(:id) => String.t(),
          required(:input) => String.t(),
          required(:target) => String.t(),
          optional(:system_prompt) => String.t(),
          optional(atom()) => any()
        }

  @type result :: %{
          required(:id) => String.t(),
          required(:input) => String.t(),
          required(:output) => String.t(),
          required(:target) => String.t(),
          optional(atom()) => any()
        }

  @type scored_result :: %{
          required(:id) => String.t(),
          required(:score) => float(),
          optional(atom()) => any()
        }

  @doc """
  Runs evaluation on a list of samples.

  ## Parameters

    * `samples` - List of sample maps with `:id`, `:input`, and `:target`
    * `config` - Configuration map with sampling parameters

  ## Returns

    * `{:ok, results}` - List of result maps with outputs
    * `{:error, reason}` - If evaluation fails
  """
  @spec run([sample()], map()) :: {:ok, [result()]} | {:error, term()}
  def run(samples, config) do
    Logger.info("Running evaluation on #{length(samples)} samples...")

    results =
      samples
      |> Enum.with_index()
      |> Enum.map(fn {sample, idx} ->
        if rem(idx + 1, 10) == 0 do
          Logger.info("Processing sample #{idx + 1}/#{length(samples)}")
        end

        case run_sample(sample, config) do
          {:ok, result} ->
            result

          {:error, reason} ->
            Logger.warning("Sample #{sample.id} failed: #{inspect(reason)}")

            %{
              id: sample.id,
              input: sample.input,
              output: "",
              target: sample.target,
              error: inspect(reason)
            }
        end
      end)

    Logger.info("Evaluation complete. Processed #{length(results)} samples.")

    {:ok, results}
  end

  @doc """
  Runs evaluation on a single sample.

  ## Parameters

    * `sample` - Sample map with `:id`, `:input`, and `:target`
    * `config` - Configuration map with sampling parameters

  ## Returns

    * `{:ok, result}` - Result map with generated output
    * `{:error, reason}` - If generation fails
  """
  @spec run_sample(sample(), map()) :: {:ok, result()} | {:error, term()}
  def run_sample(sample, config) do
    messages = create_messages(sample)

    case TinkexGenerate.generate(messages, config) do
      {:ok, response} ->
        result = %{
          id: sample.id,
          input: sample.input,
          output: response.content,
          target: sample.target
        }

        {:ok, result}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Creates messages from a sample for the LLM.

  If the sample has a `:system_prompt`, it's added as a system message.
  The `:input` field becomes the user message.
  """
  @spec create_messages(sample()) :: [Types.Message.t()]
  def create_messages(sample) do
    messages = []

    messages =
      if sample[:system_prompt] do
        [Types.message("system", sample.system_prompt) | messages]
      else
        messages
      end

    messages = messages ++ [Types.message("user", sample.input)]

    messages
  end

  @doc """
  Scores results using the specified scoring method.

  ## Scoring Methods

    * `:exact_match` - Output must exactly match target
    * `:contains` - Output must contain the target
    * `:case_insensitive` - Case-insensitive exact match

  ## Returns

  List of scored results with `:score` field (0.0 or 1.0).
  """
  @spec score_results([result()], atom()) :: [scored_result()]
  def score_results(results, method \\ :exact_match) do
    Enum.map(results, fn result ->
      score = compute_score(result.output, result.target, method)
      Map.put(result, :score, score)
    end)
  end

  @doc """
  Computes aggregate metrics from scored results.

  ## Returns

  Map with:
    * `:accuracy` - Fraction of correct results
    * `:total` - Total number of results
    * `:correct` - Number of correct results
  """
  @spec compute_metrics([scored_result()]) :: map()
  def compute_metrics(scored_results) do
    total = length(scored_results)

    if total == 0 do
      %{accuracy: 0.0, total: 0, correct: 0}
    else
      correct = Enum.count(scored_results, fn r -> r.score == 1.0 end)
      accuracy = correct / total

      %{
        accuracy: accuracy,
        total: total,
        correct: correct
      }
    end
  end

  @doc """
  Runs evaluation using an EvalEx Task.

  ## Parameters

    * `task` - EvalEx.Task struct
    * `config` - Configuration map with sampling parameters

  ## Returns

    * `{:ok, %{results: results, metrics: metrics}}`
    * `{:error, reason}`
  """
  @spec run_task(EvalEx.Task.t(), map()) :: {:ok, map()} | {:error, term()}
  def run_task(%EvalEx.Task{} = task, config) do
    Logger.info("Running task: #{task.name}")

    samples =
      case task.dataset do
        samples when is_list(samples) ->
          samples

        dataset_atom when is_atom(dataset_atom) ->
          # Load from dataset registry
          {:ok, loaded} = load_dataset(dataset_atom)
          loaded
      end

    {:ok, results} = run(samples, config)
    scored = score_results(results)
    metrics = compute_metrics(scored)

    {:ok,
     %{
       task_id: task.id,
       task_name: task.name,
       results: scored,
       metrics: metrics
     }}
  end

  # Private helpers

  defp compute_score(output, target, :exact_match) do
    if String.trim(output) == String.trim(target), do: 1.0, else: 0.0
  end

  defp compute_score(output, target, :contains) do
    if String.contains?(output, target), do: 1.0, else: 0.0
  end

  defp compute_score(output, target, :case_insensitive) do
    if String.downcase(String.trim(output)) == String.downcase(String.trim(target)) do
      1.0
    else
      0.0
    end
  end

  defp load_dataset(dataset_atom) do
    # Placeholder for dataset loading
    # In a full implementation, this would use CrucibleDatasets or HfDatasetsEx
    Logger.warning("Dataset loading not implemented for #{dataset_atom}")
    {:ok, []}
  end
end
