defmodule TinkexCookbook.Types.EncodedTextChunk do
  @moduledoc """
  A chunk of encoded text tokens.

  Mirrors `tinker.types.EncodedTextChunk` from the Python SDK.
  """

  @type t :: %__MODULE__{
          tokens: [non_neg_integer()]
        }

  @enforce_keys [:tokens]
  defstruct [:tokens]

  @doc "Creates a new EncodedTextChunk with the given tokens."
  @spec new([non_neg_integer()]) :: t()
  def new(tokens) when is_list(tokens) do
    %__MODULE__{tokens: tokens}
  end

  @doc "Returns the number of tokens in the chunk."
  @spec length(t()) :: non_neg_integer()
  def length(%__MODULE__{tokens: tokens}), do: Kernel.length(tokens)
end

defmodule TinkexCookbook.Types.ImageChunk do
  @moduledoc """
  A chunk representing image data.

  Mirrors `tinker.types.ImageChunk` from the Python SDK.
  """

  @type t :: %__MODULE__{
          data: binary(),
          format: String.t(),
          expected_tokens: non_neg_integer()
        }

  @enforce_keys [:data, :format, :expected_tokens]
  defstruct [:data, :format, :expected_tokens]

  @doc "Creates a new ImageChunk."
  @spec new(binary(), String.t(), non_neg_integer()) :: t()
  def new(data, format, expected_tokens) do
    %__MODULE__{
      data: data,
      format: format,
      expected_tokens: expected_tokens
    }
  end

  @doc "Returns the expected number of tokens for this image."
  @spec length(t()) :: non_neg_integer()
  def length(%__MODULE__{expected_tokens: expected_tokens}), do: expected_tokens
end

defmodule TinkexCookbook.Types.ModelInput do
  @moduledoc """
  Container for model input consisting of a sequence of chunks.

  Mirrors `tinker.ModelInput` from the Python SDK. A ModelInput contains
  a list of chunks which can be either EncodedTextChunks or ImageChunks.
  """

  alias TinkexCookbook.Types.{EncodedTextChunk, ImageChunk}

  @type chunk :: EncodedTextChunk.t() | ImageChunk.t()

  @type t :: %__MODULE__{
          chunks: [chunk()]
        }

  @enforce_keys [:chunks]
  defstruct [:chunks]

  @doc "Creates a new ModelInput with the given chunks."
  @spec new([chunk()]) :: t()
  def new(chunks) when is_list(chunks) do
    %__MODULE__{chunks: chunks}
  end

  @doc "Creates an empty ModelInput."
  @spec empty() :: t()
  def empty do
    %__MODULE__{chunks: []}
  end

  @doc "Creates a ModelInput from a list of token integers."
  @spec from_ints([non_neg_integer()]) :: t()
  def from_ints(tokens) when is_list(tokens) do
    new([EncodedTextChunk.new(tokens)])
  end

  @doc """
  Appends a single token to the ModelInput.

  If the last chunk is a text chunk, it is extended in place; otherwise a new
  text chunk is appended.
  """
  @spec append_int(t(), non_neg_integer()) :: t()
  def append_int(%__MODULE__{chunks: []}, token) when is_integer(token) do
    new([EncodedTextChunk.new([token])])
  end

  def append_int(%__MODULE__{chunks: chunks}, token) when is_integer(token) do
    case List.last(chunks) do
      %EncodedTextChunk{tokens: tokens} = last ->
        updated = %{last | tokens: tokens ++ [token]}
        %__MODULE__{chunks: List.replace_at(chunks, -1, updated)}

      _ ->
        %__MODULE__{chunks: chunks ++ [EncodedTextChunk.new([token])]}
    end
  end

  @doc "Returns the total length across all chunks."
  @spec length(t()) :: non_neg_integer()
  def length(%__MODULE__{chunks: chunks}) do
    Enum.reduce(chunks, 0, fn chunk, acc ->
      acc + chunk_length(chunk)
    end)
  end

  @doc """
  Extracts all tokens from the model input.

  For text chunks, returns the actual tokens.
  For image chunks, returns placeholder zeros.
  """
  @spec all_tokens(t()) :: [non_neg_integer()]
  def all_tokens(%__MODULE__{chunks: chunks}) do
    Enum.flat_map(chunks, fn
      %EncodedTextChunk{tokens: tokens} -> tokens
      %ImageChunk{expected_tokens: n} -> List.duplicate(0, n)
    end)
  end

  defp chunk_length(%EncodedTextChunk{} = chunk), do: EncodedTextChunk.length(chunk)
  defp chunk_length(%ImageChunk{} = chunk), do: ImageChunk.length(chunk)
end
