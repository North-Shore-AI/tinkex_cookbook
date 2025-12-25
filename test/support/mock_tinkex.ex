defmodule TinkexCookbook.Test.MockTinkex do
  @moduledoc """
  Mock Tinkex clients for testing training and sampling without network access.

  These mocks provide deterministic responses that match the Tinkex API structure.
  """

  alias TinkexCookbook.Test.MockTinkex.SamplingClient

  defmodule TrainingClient do
    @moduledoc """
    Mock training client that returns predictable training outputs.
    """

    defstruct [:step, :model_name]

    @type t :: %__MODULE__{
            step: non_neg_integer(),
            model_name: String.t()
          }

    @doc "Creates a new mock training client."
    @spec new(keyword()) :: t()
    def new(opts \\ []) do
      %__MODULE__{
        step: Keyword.get(opts, :step, 0),
        model_name: Keyword.get(opts, :model_name, "mock-model")
      }
    end

    @doc """
    Mock forward_backward that returns deterministic loss metrics.
    """
    @spec forward_backward(t(), list(), atom() | String.t()) :: {:ok, Task.t()}
    def forward_backward(%__MODULE__{} = _client, batch, _loss_fn \\ :cross_entropy) do
      loss = 0.5 - length(batch) * 0.01

      output = %{
        loss_fn_output_type: "cross_entropy",
        loss_fn_outputs:
          Enum.map(batch, fn _datum ->
            %{"logprobs" => List.duplicate(-1.0, 10)}
          end),
        metrics: %{"loss" => max(loss, 0.01)}
      }

      {:ok, Task.async(fn -> {:ok, output} end)}
    end

    @doc """
    Mock forward_backward_custom for DPO.
    """
    @spec forward_backward_custom(t(), list(), term()) :: {:ok, Task.t()}
    def forward_backward_custom(%__MODULE__{} = client, batch, loss_fn \\ :custom) do
      forward_backward(client, batch, loss_fn)
    end

    @doc """
    Mock optim_step that simulates optimizer update.
    """
    @spec optim_step(t(), map()) :: {:ok, Task.t()}
    def optim_step(%__MODULE__{} = _client, _adam_params) do
      {:ok, Task.async(fn -> {:ok, %{}} end)}
    end

    @doc """
    Mock save_weights_for_sampler that returns a checkpoint path.
    """
    @spec save_weights_for_sampler(t(), String.t()) :: {:ok, Task.t()}
    def save_weights_for_sampler(%__MODULE__{} = _client, name) do
      {:ok, Task.async(fn -> {:ok, %{path: "/tmp/mock_sampler_#{name}"}} end)}
    end

    @doc """
    Mock save_state that returns a checkpoint path.
    """
    @spec save_state(t(), String.t()) :: {:ok, Task.t()}
    def save_state(%__MODULE__{} = _client, name) do
      {:ok, Task.async(fn -> {:ok, %{path: "/tmp/mock_state_#{name}"}} end)}
    end

    @doc """
    Mock load_state that returns a response map.
    """
    @spec load_state(t(), String.t()) :: {:ok, Task.t()}
    def load_state(%__MODULE__{} = _client, _path) do
      {:ok, Task.async(fn -> {:ok, %{}} end)}
    end

    @doc """
    Mock load_state_with_optimizer that returns a response map.
    """
    @spec load_state_with_optimizer(t(), String.t()) :: {:ok, Task.t()}
    def load_state_with_optimizer(%__MODULE__{} = _client, _path) do
      {:ok, Task.async(fn -> {:ok, %{}} end)}
    end

    @doc """
    Mock save_weights_and_get_sampling_client that returns a sampling client.
    """
    @spec save_weights_and_get_sampling_client(t(), keyword()) :: {:ok, Task.t()}
    def save_weights_and_get_sampling_client(%__MODULE__{} = _client, _opts \\ []) do
      {:ok, Task.async(fn -> {:ok, SamplingClient.new()} end)}
    end

    @doc """
    Mock create_sampling_client_async that returns a sampling client.
    """
    @spec create_sampling_client_async(t(), String.t()) :: Task.t()
    def create_sampling_client_async(%__MODULE__{} = _client, _path) do
      Task.async(fn -> {:ok, SamplingClient.new()} end)
    end
  end

  defmodule SamplingClient do
    @moduledoc """
    Mock sampling client that returns predictable responses.
    """

    defstruct [:model_name, :response_tokens]

    @type t :: %__MODULE__{
            model_name: String.t(),
            response_tokens: [non_neg_integer()]
          }

    @doc "Creates a new mock sampling client."
    @spec new(keyword()) :: t()
    def new(opts \\ []) do
      # Default response tokens spell "Hello!" in ASCII
      default_tokens = String.to_charlist("Hello!")

      %__MODULE__{
        model_name: Keyword.get(opts, :model_name, "mock-model"),
        response_tokens: Keyword.get(opts, :response_tokens, default_tokens)
      }
    end

    @doc """
    Mock sample that returns deterministic token sequences.
    """
    @spec sample(t(), any(), map(), keyword()) :: {:ok, Task.t()}
    def sample(%__MODULE__{response_tokens: tokens} = _client, _model_input, _params, _opts \\ []) do
      response = %{
        sequences: [
          %{
            tokens: tokens,
            text: List.to_string(tokens),
            stop_reason: "end_of_turn",
            logprobs: List.duplicate(-1.0, length(tokens))
          }
        ],
        prompt_logprobs: nil,
        type: "sample"
      }

      {:ok, Task.async(fn -> {:ok, response} end)}
    end

    @doc """
    Mock compute_logprobs that returns a list of logprobs aligned to input length.
    """
    @spec compute_logprobs(t(), any(), keyword()) :: {:ok, Task.t()}
    def compute_logprobs(%__MODULE__{} = _client, model_input, _opts \\ []) do
      length =
        case model_input do
          %Tinkex.Types.ModelInput{chunks: chunks} ->
            Enum.reduce(chunks, 0, fn
              %Tinkex.Types.EncodedTextChunk{tokens: tokens}, acc -> acc + length(tokens)
              %Tinkex.Types.ImageChunk{expected_tokens: n}, acc -> acc + (n || 0)
              _chunk, acc -> acc
            end)

          _ ->
            0
        end

      {:ok, Task.async(fn -> {:ok, List.duplicate(-1.0, length)} end)}
    end
  end

  defmodule ServiceClient do
    @moduledoc """
    Mock service client for creating training and sampling clients.
    """

    defstruct [:base_url]

    @type t :: %__MODULE__{
            base_url: String.t() | nil
          }

    @doc "Creates a new mock service client."
    @spec new(keyword()) :: t()
    def new(opts \\ []) do
      %__MODULE__{
        base_url: Keyword.get(opts, :base_url)
      }
    end

    @doc """
    Creates a mock training client with LoRA configuration.
    """
    @spec create_lora_training_client(t(), keyword()) :: {:ok, TrainingClient.t()}
    def create_lora_training_client(%__MODULE__{} = _client, opts \\ []) do
      {:ok, TrainingClient.new(opts)}
    end

    @doc """
    Async mock training client creation.
    """
    @spec create_lora_training_client_async(t(), String.t(), keyword()) :: Task.t()
    def create_lora_training_client_async(%__MODULE__{} = _client, _model, opts \\ []) do
      Task.async(fn -> {:ok, TrainingClient.new(opts)} end)
    end

    @doc """
    Creates a mock sampling client.
    """
    @spec create_sampling_client(t(), keyword()) :: {:ok, SamplingClient.t()}
    def create_sampling_client(%__MODULE__{} = _client, opts \\ []) do
      {:ok, SamplingClient.new(opts)}
    end

    @doc """
    Async mock sampling client creation.
    """
    @spec create_sampling_client_async(t(), keyword()) :: Task.t()
    def create_sampling_client_async(%__MODULE__{} = _client, _opts \\ []) do
      Task.async(fn -> {:ok, SamplingClient.new()} end)
    end

    @doc """
    Async mock training client creation from state.
    """
    @spec create_training_client_from_state_async(t(), String.t(), keyword()) :: Task.t()
    def create_training_client_from_state_async(%__MODULE__{} = _client, _path, _opts \\ []) do
      Task.async(fn -> {:ok, TrainingClient.new()} end)
    end

    @doc """
    Async mock training client creation from state with optimizer.
    """
    @spec create_training_client_from_state_with_optimizer_async(t(), String.t(), keyword()) ::
            Task.t()
    def create_training_client_from_state_with_optimizer_async(
          %__MODULE__{} = _client,
          _path,
          _opts \\ []
        ) do
      Task.async(fn -> {:ok, TrainingClient.new()} end)
    end
  end
end
