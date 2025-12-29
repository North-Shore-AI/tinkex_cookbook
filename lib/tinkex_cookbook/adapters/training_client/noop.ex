defmodule TinkexCookbook.Adapters.TrainingClient.Noop do
  @moduledoc """
  No-op TrainingClient adapter for testing.

  This adapter implements the CrucibleTrain.Ports.TrainingClient behaviour
  but does nothing. Useful for testing recipes without actual training.
  """

  @behaviour CrucibleTrain.Ports.TrainingClient

  @type session :: %{id: reference(), model: String.t(), config: map()}
  @type future :: {:noop_future, reference()}

  @impl true
  def start_session(_adapter_opts, config) do
    model = Map.get(config, :model, "test-model")

    {:ok,
     %{
       id: make_ref(),
       model: model,
       config: config
     }}
  end

  @impl true
  def forward_backward(_adapter_opts, _session, datums) do
    # Return a mock future
    {:noop_future,
     %{
       loss_fn_outputs:
         Enum.map(datums, fn _ ->
           %{"logprobs" => %{"data" => []}}
         end)
     }}
  end

  @impl true
  def optim_step(_adapter_opts, _session, _lr) do
    {:noop_future, :ok}
  end

  @impl true
  def await(_adapter_opts, {:noop_future, result}) do
    {:ok, result}
  end

  @impl true
  def save_checkpoint(_adapter_opts, _session, _path) do
    :ok
  end

  @impl true
  def load_checkpoint(_adapter_opts, _session, _path) do
    :ok
  end

  @impl true
  def close_session(_adapter_opts, _session) do
    :ok
  end
end
