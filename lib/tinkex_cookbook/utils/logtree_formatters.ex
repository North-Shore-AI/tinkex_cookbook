defmodule TinkexCookbook.Utils.LogtreeFormatters do
  @moduledoc """
  Minimal formatter helpers for logtree output.
  """

  defmodule ConversationFormatter do
    @moduledoc """
    Formatter for conversation messages.
    """

    defstruct [:messages]

    @type t :: %__MODULE__{messages: list(map())}

    @spec to_html(t()) :: String.t()
    def to_html(%__MODULE__{messages: messages}) do
      parts =
        Enum.map(messages, fn msg ->
          role = Map.get(msg, :role) || Map.get(msg, "role") || ""
          content = Map.get(msg, :content) || Map.get(msg, "content") || ""

          "<div class=\"lt-message lt-message-#{role}\">" <>
            "<span class=\"lt-message-role\">#{role}:</span>" <>
            "<span class=\"lt-message-content\">#{content}</span>" <>
            "</div>"
        end)

      ["<div class=\"lt-conversation\">", Enum.join(parts, "\n"), "</div>"]
      |> Enum.join("\n")
    end

    @spec get_css(t()) :: String.t()
    def get_css(_formatter) do
      """
      .lt-conversation { display: flex; flex-direction: column; gap: 0.5rem; }
      .lt-message { padding: 0.75rem; border-radius: 6px; }
      .lt-message-role { font-weight: 600; margin-right: 0.5rem; }
      .lt-message-content { white-space: pre-wrap; }
      """
    end
  end
end
