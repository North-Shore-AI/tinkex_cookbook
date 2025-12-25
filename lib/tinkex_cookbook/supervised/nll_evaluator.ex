defmodule TinkexCookbook.Supervised.NLLEvaluator do
  @moduledoc """
  Evaluator that computes negative log-likelihood on a test dataset.

  This evaluator runs forward passes on a set of datums and computes the
  weighted mean NLL across all samples.

  ## Usage

      # From a list of datums
      evaluator = NLLEvaluator.new(test_datums, name: "test")
      {:ok, metrics} = NLLEvaluator.evaluate(evaluator, training_client)
      # => %{"test/nll" => 2.345}

      # From a dataset
      evaluator = NLLEvaluator.from_dataset(test_dataset, name: "validation")

  """

  @behaviour TinkexCookbook.Eval.Evaluators.TrainingClientEvaluator

  alias TinkexCookbook.Supervised.{Common, SupervisedDataset}
  alias TinkexCookbook.Types.{Datum, TensorData}

  defstruct [:data, :name]

  @type t :: %__MODULE__{
          data: [Datum.t()],
          name: String.t()
        }

  @doc """
  Creates a new NLLEvaluator from a list of datums.

  ## Options

  - `:name` - The name prefix for metrics (default: "test")
  """
  @spec new([Datum.t()], keyword()) :: t()
  def new(data, opts \\ []) do
    name = Keyword.get(opts, :name, "test")
    %__MODULE__{data: data, name: name}
  end

  @doc """
  Creates an NLLEvaluator from a SupervisedDataset.

  Collects all batches from the dataset into a single datum list.
  """
  @spec from_dataset(SupervisedDataset.t(), keyword()) :: t()
  def from_dataset(dataset, opts \\ []) do
    # Collect all datums from all batches
    all_data =
      0..(SupervisedDataset.length(dataset) - 1)
      |> Enum.flat_map(fn i -> SupervisedDataset.get_batch(dataset, i) end)

    new(all_data, opts)
  end

  @doc """
  Evaluates the NLL on the stored datums using the training client.

  Returns a map with a single key "{name}/nll" containing the mean NLL value.
  """
  @impl true
  @spec evaluate(t(), pid() | struct()) :: {:ok, map()} | {:error, term()}
  def evaluate(%__MODULE__{data: data, name: name}, training_client) do
    case do_forward(training_client, data) do
      {:ok, result} ->
        logprobs = extract_logprobs(result)
        weights = extract_weights(data)
        nll = Common.compute_mean_nll(logprobs, weights)
        key = "#{name}/nll"
        {:ok, %{key => nll}}

      {:error, _} = error ->
        error
    end
  end

  # Forward pass using Tinkex training client
  defp do_forward(training_client, datums) do
    # Convert datums to the format expected by Tinkex
    tinkex_datums = Enum.map(datums, &datum_to_tinkex/1)

    # Use TrainingClient.forward/4 (client, data, loss_fn, opts)
    # Always returns {:ok, task}
    {:ok, task} =
      Tinkex.TrainingClient.forward(training_client, tinkex_datums, :cross_entropy, [])

    # Wait for the async task to complete
    Task.await(task, :infinity)
  end

  # Extract logprobs from forward result
  defp extract_logprobs(result) do
    result.loss_fn_outputs
    |> Enum.map(fn output ->
      logprobs_data = output["logprobs"]
      # Convert to TensorData if needed
      if is_struct(logprobs_data, TensorData) do
        logprobs_data
      else
        TensorData.from_list(logprobs_data, :float32)
      end
    end)
  end

  # Extract weights from datums
  defp extract_weights(datums) do
    Enum.map(datums, fn datum ->
      Datum.get_weights(datum)
    end)
  end

  # Convert internal Datum to Tinkex format
  defp datum_to_tinkex(%Datum{} = datum) do
    %{
      model_input: model_input_to_tinkex(datum.model_input),
      loss_fn_inputs: loss_fn_inputs_to_tinkex(datum.loss_fn_inputs)
    }
  end

  defp model_input_to_tinkex(model_input) do
    tokens =
      model_input.chunks
      |> Enum.flat_map(fn
        %TinkexCookbook.Types.EncodedTextChunk{tokens: tokens} -> tokens
        _ -> []
      end)

    %{tokens: tokens}
  end

  defp loss_fn_inputs_to_tinkex(loss_fn_inputs) do
    Map.new(loss_fn_inputs, fn {key, tensor_data} ->
      {key,
       %{
         data: TensorData.to_list(tensor_data),
         dtype: tensor_data.dtype,
         shape: tensor_data.shape
       }}
    end)
  end
end
