defmodule TinkexCookbook.Renderers.Types do
  @moduledoc """
  Type definitions for the renderer system.

  This module defines the core types used throughout the rendering pipeline:

  - `TextPart` - A text content part
  - `ImagePart` - An image content part
  - `Message` - A conversation message
  - `ToolCall` - A tool/function call request
  - `RenderedMessage` - A message rendered into token chunks
  """

  # =============================================================================
  # Content Parts
  # =============================================================================

  defmodule TextPart do
    @moduledoc """
    A text content part in a multimodal message.
    """
    @enforce_keys [:type, :text]
    defstruct [:type, :text]

    @type t :: %__MODULE__{
            type: String.t(),
            text: String.t()
          }
  end

  defmodule ImagePart do
    @moduledoc """
    An image content part in a multimodal message.

    The `image` field can be:
    - A URL string (e.g., "https://example.com/image.jpg")
    - A data URI string (e.g., "data:image/jpeg;base64,...")
    - Binary image data
    """
    @enforce_keys [:type, :image]
    defstruct [:type, :image]

    @type t :: %__MODULE__{
            type: String.t(),
            image: String.t() | binary()
          }
  end

  @type content_part :: TextPart.t() | ImagePart.t()
  @type content :: String.t() | [content_part()]

  # =============================================================================
  # Tool Types
  # =============================================================================

  defmodule FunctionBody do
    @moduledoc """
    The function body of a tool call containing name and arguments.
    """
    @enforce_keys [:name, :arguments]
    defstruct [:name, :arguments]

    @type t :: %__MODULE__{
            name: String.t(),
            arguments: String.t()
          }
  end

  defmodule ToolCall do
    @moduledoc """
    A tool/function call following the OpenAI format.
    """
    @enforce_keys [:type, :function]
    defstruct [:type, :function, :id]

    @type t :: %__MODULE__{
            type: String.t(),
            function: FunctionBody.t(),
            id: String.t() | nil
          }
  end

  defmodule ToolOk do
    @moduledoc """
    Successful tool execution result.
    """
    defstruct output: "", message: "", brief: ""

    @type t :: %__MODULE__{
            output: String.t(),
            message: String.t(),
            brief: String.t()
          }
  end

  defmodule ToolError do
    @moduledoc """
    Tool execution error result.
    """
    defstruct output: "", message: "", brief: ""

    @type t :: %__MODULE__{
            output: String.t(),
            message: String.t(),
            brief: String.t()
          }
  end

  defmodule ToolResult do
    @moduledoc """
    Complete tool execution result with tracking ID.
    """
    defstruct [:tool_call_id, :result]

    @type t :: %__MODULE__{
            tool_call_id: String.t() | nil,
            result: ToolOk.t() | ToolError.t()
          }
  end

  # =============================================================================
  # Message Types
  # =============================================================================

  defmodule Message do
    @moduledoc """
    A single message in a conversation.

    ## Fields

    - `role` - The role of the message sender (e.g., "user", "assistant", "system", "tool")
    - `content` - The message content, either a string or a list of content parts
    - `tool_calls` - Optional list of tool calls made by the assistant
    - `thinking` - Optional thinking/reasoning content
    - `trainable` - Optional flag for customized training loss
    - `tool_call_id` - Optional ID for tool response messages
    - `name` - Optional name for the message sender
    """
    @enforce_keys [:role, :content]
    defstruct [:role, :content, :tool_calls, :thinking, :trainable, :tool_call_id, :name]

    @type t :: %__MODULE__{
            role: String.t(),
            content: TinkexCookbook.Renderers.Types.content(),
            tool_calls: [TinkexCookbook.Renderers.Types.ToolCall.t()] | nil,
            thinking: String.t() | nil,
            trainable: boolean() | nil,
            tool_call_id: String.t() | nil,
            name: String.t() | nil
          }
  end

  # =============================================================================
  # Rendered Message
  # =============================================================================

  defmodule RenderedMessage do
    @moduledoc """
    A rendered message containing token chunks for training/sampling.

    ## Fields

    - `prefix` - Optional message header (e.g., role tokens)
    - `content` - List of content chunks (text/image)
    - `suffix` - Optional message footer (e.g., stop tokens)
    """
    @enforce_keys [:content]
    defstruct [:prefix, :content, :suffix]

    @type chunk :: map()

    @type t :: %__MODULE__{
            prefix: chunk() | nil,
            content: [chunk()],
            suffix: chunk() | nil
          }
  end

  # =============================================================================
  # Constructor Functions
  # =============================================================================

  @doc """
  Creates a text content part.

  ## Examples

      iex> TinkexCookbook.Renderers.Types.text_part("Hello!")
      %TinkexCookbook.Renderers.Types.TextPart{type: "text", text: "Hello!"}
  """
  @spec text_part(String.t()) :: TextPart.t()
  def text_part(text) when is_binary(text) do
    %TextPart{type: "text", text: text}
  end

  @doc """
  Creates an image content part.

  ## Examples

      iex> TinkexCookbook.Renderers.Types.image_part("https://example.com/img.jpg")
      %TinkexCookbook.Renderers.Types.ImagePart{type: "image", image: "https://example.com/img.jpg"}
  """
  @spec image_part(String.t() | binary()) :: ImagePart.t()
  def image_part(image) do
    %ImagePart{type: "image", image: image}
  end

  @doc """
  Creates a conversation message.

  ## Options

  - `:tool_calls` - List of tool calls
  - `:thinking` - Thinking/reasoning content
  - `:trainable` - Whether this message should contribute to training loss
  - `:tool_call_id` - ID for tool response messages
  - `:name` - Name of the message sender

  ## Examples

      iex> TinkexCookbook.Renderers.Types.message("user", "Hello!")
      %TinkexCookbook.Renderers.Types.Message{role: "user", content: "Hello!", tool_calls: nil, thinking: nil, trainable: nil, tool_call_id: nil, name: nil}
  """
  @spec message(String.t(), content(), keyword()) :: Message.t()
  def message(role, content, opts \\ []) when is_binary(role) do
    %Message{
      role: role,
      content: content,
      tool_calls: Keyword.get(opts, :tool_calls),
      thinking: Keyword.get(opts, :thinking),
      trainable: Keyword.get(opts, :trainable),
      tool_call_id: Keyword.get(opts, :tool_call_id),
      name: Keyword.get(opts, :name)
    }
  end

  @doc """
  Creates a tool call.

  ## Examples

      iex> TinkexCookbook.Renderers.Types.tool_call("search", ~s({"query": "test"}), id: "call_123")
      %TinkexCookbook.Renderers.Types.ToolCall{
        type: "function",
        function: %TinkexCookbook.Renderers.Types.FunctionBody{name: "search", arguments: ~s({"query": "test"})},
        id: "call_123"
      }
  """
  @spec tool_call(String.t(), String.t(), keyword()) :: ToolCall.t()
  def tool_call(name, arguments, opts \\ []) when is_binary(name) and is_binary(arguments) do
    %ToolCall{
      type: "function",
      function: %FunctionBody{name: name, arguments: arguments},
      id: Keyword.get(opts, :id)
    }
  end

  # =============================================================================
  # Helper Functions
  # =============================================================================

  @doc """
  Ensures content is text-only and returns it as a string.

  Raises `ArgumentError` if content contains images or multiple parts.

  ## Examples

      iex> TinkexCookbook.Renderers.Types.ensure_text("Hello")
      "Hello"

      iex> TinkexCookbook.Renderers.Types.ensure_text([%TinkexCookbook.Renderers.Types.TextPart{type: "text", text: "Hello"}])
      "Hello"
  """
  @spec ensure_text(content()) :: String.t()
  def ensure_text(content) when is_binary(content), do: content

  def ensure_text([%TextPart{text: text}]), do: text

  def ensure_text(content) when is_list(content) do
    raise ArgumentError,
          "Expected text content, got multimodal content with #{length(content)} parts"
  end
end
