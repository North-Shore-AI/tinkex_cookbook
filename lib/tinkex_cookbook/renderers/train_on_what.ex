defmodule TinkexCookbook.Renderers.TrainOnWhat do
  @moduledoc """
  Enum representing what parts of a conversation should be used for training loss.

  This module mirrors the Python `TrainOnWhat` enum from `tinker_cookbook.renderers`.
  The values are kept as strings to match the Python behavior exactly.

  ## Values

  - `"last_assistant_message"` - Only the last assistant message contributes to loss
  - `"all_assistant_messages"` - All assistant messages contribute to loss
  - `"all_messages"` - All messages (user, assistant, system) contribute to loss
  - `"all_tokens"` - All tokens including role prefixes contribute to loss
  - `"all_user_and_system_messages"` - User and system messages contribute to loss
  - `"customized"` - Each message has a `trainable` field controlling loss contribution

  ## Examples

      iex> TinkexCookbook.Renderers.TrainOnWhat.all_assistant_messages()
      "all_assistant_messages"

      iex> TinkexCookbook.Renderers.TrainOnWhat.valid?("all_tokens")
      true

      iex> TinkexCookbook.Renderers.TrainOnWhat.valid?("invalid")
      false
  """

  @type t :: String.t()

  @last_assistant_message "last_assistant_message"
  @all_assistant_messages "all_assistant_messages"
  @all_messages "all_messages"
  @all_tokens "all_tokens"
  @all_user_and_system_messages "all_user_and_system_messages"
  @customized "customized"

  @all_values [
    @last_assistant_message,
    @all_assistant_messages,
    @all_messages,
    @all_tokens,
    @all_user_and_system_messages,
    @customized
  ]

  @doc """
  Returns the value for training on only the last assistant message.
  """
  @spec last_assistant_message() :: t()
  def last_assistant_message, do: @last_assistant_message

  @doc """
  Returns the value for training on all assistant messages.
  """
  @spec all_assistant_messages() :: t()
  def all_assistant_messages, do: @all_assistant_messages

  @doc """
  Returns the value for training on all messages.
  """
  @spec all_messages() :: t()
  def all_messages, do: @all_messages

  @doc """
  Returns the value for training on all tokens.
  """
  @spec all_tokens() :: t()
  def all_tokens, do: @all_tokens

  @doc """
  Returns the value for training on all user and system messages.
  """
  @spec all_user_and_system_messages() :: t()
  def all_user_and_system_messages, do: @all_user_and_system_messages

  @doc """
  Returns the value for customized per-message training.
  """
  @spec customized() :: t()
  def customized, do: @customized

  @doc """
  Returns all valid TrainOnWhat values.
  """
  @spec values() :: [t()]
  def values, do: @all_values

  @doc """
  Checks if a value is a valid TrainOnWhat value.
  """
  @spec valid?(term()) :: boolean()
  def valid?(value) when value in @all_values, do: true
  def valid?(_), do: false

  @doc """
  Parses a string into a TrainOnWhat value.

  Returns `{:ok, value}` if valid, `{:error, reason}` otherwise.
  """
  @spec from_string(String.t()) :: {:ok, t()} | {:error, String.t()}
  def from_string(value) when value in @all_values, do: {:ok, value}

  def from_string(value) when is_binary(value) do
    {:error,
     "Invalid TrainOnWhat value: #{inspect(value)}. Valid values: #{inspect(@all_values)}"}
  end

  def from_string(value) do
    {:error, "Expected string, got: #{inspect(value)}"}
  end
end
