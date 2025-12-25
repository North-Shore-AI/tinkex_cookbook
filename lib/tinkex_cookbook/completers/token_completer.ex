defmodule TinkexCookbook.Completers.TokensWithLogprobs do
  @moduledoc """
  Token completion result with optional logprobs.
  """

  @type t :: %__MODULE__{
          tokens: [integer()],
          maybe_logprobs: [float()] | nil
        }

  defstruct [:tokens, :maybe_logprobs]

  @spec logprobs!(t()) :: [float()]
  def logprobs!(%__MODULE__{maybe_logprobs: nil}) do
    raise ArgumentError, "Logprobs are not available"
  end

  def logprobs!(%__MODULE__{maybe_logprobs: logprobs}), do: logprobs
end

defmodule TinkexCookbook.Completers.TokenCompleter do
  @moduledoc """
  Behaviour for token-based completers.
  """

  alias TinkexCookbook.Completers.TokensWithLogprobs
  alias TinkexCookbook.Types.ModelInput

  @type stop_condition :: [String.t()] | [integer()]

  @callback complete(struct(), ModelInput.t(), stop_condition()) ::
              {:ok, TokensWithLogprobs.t()} | {:error, term()}
end
