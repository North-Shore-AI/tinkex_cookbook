defmodule TinkexCookbook.Ports do
  @moduledoc """
  Composition root for cookbook ports and adapters.

  Build a `Ports` struct once (per recipe or per process), then pass it
  to port modules (`Ports.VectorStore`, `Ports.DatasetStore`, etc.).
  """

  alias TinkexCookbook.Adapters

  @type adapter_ref :: {module(), keyword()}
  @type port_key ::
          :vector_store
          | :embedding_client
          | :llm_client
          | :dataset_store
          | :hub_client
          | :blob_store

  @type t :: %__MODULE__{
          vector_store: adapter_ref(),
          embedding_client: adapter_ref(),
          llm_client: adapter_ref(),
          dataset_store: adapter_ref(),
          hub_client: adapter_ref(),
          blob_store: adapter_ref()
        }

  @default_ports %{
    vector_store: {Adapters.VectorStore.Noop, []},
    embedding_client: {Adapters.EmbeddingClient.Noop, []},
    llm_client: {Adapters.LLMClient.Noop, []},
    dataset_store: {Adapters.DatasetStore.HfDatasets, []},
    hub_client: {Adapters.HubClient.HfHub, []},
    blob_store: {Adapters.BlobStore.Local, []}
  }

  @port_behaviours %{
    vector_store: TinkexCookbook.Ports.VectorStore,
    embedding_client: TinkexCookbook.Ports.EmbeddingClient,
    llm_client: TinkexCookbook.Ports.LLMClient,
    dataset_store: TinkexCookbook.Ports.DatasetStore,
    hub_client: TinkexCookbook.Ports.HubClient,
    blob_store: TinkexCookbook.Ports.BlobStore
  }

  defstruct Map.keys(@default_ports)

  @doc """
  Build a Ports struct.

  Accepts optional overrides via `:ports` (map or keyword list).
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    config = Application.get_env(:tinkex_cookbook, __MODULE__, %{})
    overrides = Keyword.get(opts, :ports, %{})

    resolved =
      @default_ports
      |> Map.merge(normalize_port_map(config))
      |> Map.merge(normalize_port_map(overrides))

    validate_ports!(resolved)
    struct!(__MODULE__, resolved)
  end

  @doc """
  Resolve a port adapter from a Ports struct.
  """
  @spec resolve(t(), port_key()) :: adapter_ref()
  def resolve(%__MODULE__{} = ports, port_key) do
    Map.fetch!(ports, port_key)
  end

  defp normalize_port_map(ports) when is_list(ports) do
    ports
    |> Enum.into(%{})
    |> normalize_port_map()
  end

  defp normalize_port_map(ports) when is_map(ports) do
    Enum.reduce(ports, %{}, fn {port_key, adapter}, acc ->
      if adapter == nil do
        acc
      else
        Map.put(acc, port_key, normalize_adapter(adapter))
      end
    end)
  end

  defp normalize_port_map(_), do: %{}

  defp normalize_adapter({module, opts}) when is_atom(module) and is_list(opts) do
    {module, opts}
  end

  defp normalize_adapter(module) when is_atom(module), do: {module, []}

  defp normalize_adapter(other) do
    raise ArgumentError, "Invalid adapter reference: #{inspect(other)}"
  end

  defp validate_ports!(ports) do
    Enum.each(ports, fn {port_key, {module, _opts}} ->
      validate_adapter!(port_key, module)
    end)
  end

  defp validate_adapter!(port_key, module) do
    behaviour = Map.fetch!(@port_behaviours, port_key)
    callbacks = behaviour.behaviour_info(:callbacks)

    unless Code.ensure_loaded?(module) do
      raise ArgumentError, "Adapter #{inspect(module)} for #{inspect(port_key)} is not available"
    end

    missing =
      Enum.reject(callbacks, fn {fun, arity} ->
        function_exported?(module, fun, arity)
      end)

    if missing != [] do
      raise ArgumentError,
            "Adapter #{inspect(module)} for #{inspect(port_key)} is missing callbacks: #{inspect(missing)}"
    end
  end
end
