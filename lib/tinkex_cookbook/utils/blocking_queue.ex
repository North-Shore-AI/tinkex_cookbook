defmodule TinkexCookbook.Utils.BlockingQueue do
  @moduledoc """
  Simple blocking queue backed by a GenServer.
  """

  use GenServer

  @type t :: pid()

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(_opts \\ []) do
    GenServer.start_link(__MODULE__, %{
      queue: :queue.new(),
      waiters: :queue.new()
    })
  end

  @spec push(t(), term()) :: :ok
  def push(pid, item) do
    GenServer.cast(pid, {:push, item})
  end

  @spec pop(t()) :: term()
  def pop(pid) do
    GenServer.call(pid, :pop, :infinity)
  end

  @impl true
  def init(state), do: {:ok, state}

  @impl true
  def handle_cast({:push, item}, %{queue: queue, waiters: waiters} = state) do
    case :queue.out(waiters) do
      {{:value, from}, rest_waiters} ->
        GenServer.reply(from, item)
        {:noreply, %{state | waiters: rest_waiters}}

      {:empty, _} ->
        {:noreply, %{state | queue: :queue.in(item, queue)}}
    end
  end

  @impl true
  def handle_call(:pop, from, %{queue: queue, waiters: waiters} = state) do
    case :queue.out(queue) do
      {{:value, item}, rest_queue} ->
        {:reply, item, %{state | queue: rest_queue}}

      {:empty, _} ->
        {:noreply, %{state | waiters: :queue.in(from, waiters)}}
    end
  end
end
