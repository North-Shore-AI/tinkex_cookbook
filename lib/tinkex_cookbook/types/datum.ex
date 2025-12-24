defmodule TinkexCookbook.Types.Datum do
  @moduledoc """
  A training datum containing model input and loss function inputs.

  Mirrors `tinker.Datum` from the Python SDK. A Datum is the fundamental
  unit passed to the training API, containing:
  - `model_input`: The input to the model (sequence of chunks)
  - `loss_fn_inputs`: A map containing "weights" and "target_tokens" TensorData
  """

  alias TinkexCookbook.Types.{ModelInput, TensorData}

  @type t :: %__MODULE__{
          model_input: ModelInput.t(),
          loss_fn_inputs: %{String.t() => TensorData.t()}
        }

  @enforce_keys [:model_input, :loss_fn_inputs]
  defstruct [:model_input, :loss_fn_inputs]

  @doc "Creates a new Datum with model_input and loss_fn_inputs."
  @spec new(ModelInput.t(), %{String.t() => TensorData.t()}) :: t()
  def new(%ModelInput{} = model_input, loss_fn_inputs) when is_map(loss_fn_inputs) do
    %__MODULE__{
      model_input: model_input,
      loss_fn_inputs: loss_fn_inputs
    }
  end

  @doc "Returns the weights TensorData from loss_fn_inputs."
  @spec get_weights(t()) :: TensorData.t() | nil
  def get_weights(%__MODULE__{loss_fn_inputs: inputs}) do
    Map.get(inputs, "weights")
  end

  @doc "Returns the target_tokens TensorData from loss_fn_inputs."
  @spec get_target_tokens(t()) :: TensorData.t() | nil
  def get_target_tokens(%__MODULE__{loss_fn_inputs: inputs}) do
    Map.get(inputs, "target_tokens")
  end

  @doc "Returns the sum of weights (number of loss tokens)."
  @spec num_loss_tokens(t()) :: number()
  def num_loss_tokens(%__MODULE__{} = datum) do
    case get_weights(datum) do
      nil -> 0
      weights -> TensorData.sum(weights)
    end
  end
end
