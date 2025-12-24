defmodule TinkexCookbook.Test.MockTinkex do
  @moduledoc """
  Mock Tinkex clients for testing training and sampling without network access.

  These mocks provide deterministic responses that match the Tinkex API structure.
  """

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
    @spec forward_backward(t(), list()) :: {:ok, map()}
    def forward_backward(%__MODULE__{} = _client, batch) do
      loss = 0.5 - length(batch) * 0.01

      output = %{
        loss_fn_output_type: "cross_entropy",
        loss_fn_outputs:
          Enum.map(batch, fn _datum ->
            %{"logprobs" => List.duplicate(-1.0, 10)}
          end),
        metrics: %{"loss" => max(loss, 0.01)}
      }

      {:ok, output}
    end

    @doc """
    Mock optim_step that simulates optimizer update.
    """
    @spec optim_step(t()) :: :ok
    def optim_step(%__MODULE__{} = _client) do
      :ok
    end

    @doc """
    Mock optim_step with adam params.
    """
    @spec optim_step(t(), map()) :: :ok
    def optim_step(%__MODULE__{} = _client, _adam_params) do
      :ok
    end

    @doc """
    Mock save_weights that returns a checkpoint path.
    """
    @spec save_weights(t(), String.t()) :: {:ok, String.t()}
    def save_weights(%__MODULE__{} = _client, name) do
      {:ok, "/tmp/mock_checkpoint_#{name}"}
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
    @spec sample(t(), any(), keyword()) :: {:ok, map()}
    def sample(%__MODULE__{response_tokens: tokens} = _client, _model_input, _opts \\ []) do
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

      {:ok, response}
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
    Creates a mock sampling client.
    """
    @spec create_sampling_client(t(), keyword()) :: {:ok, SamplingClient.t()}
    def create_sampling_client(%__MODULE__{} = _client, opts \\ []) do
      {:ok, SamplingClient.new(opts)}
    end
  end
end
