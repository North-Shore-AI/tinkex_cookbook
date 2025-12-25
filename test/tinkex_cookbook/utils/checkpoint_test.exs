defmodule TinkexCookbook.Utils.CheckpointTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Utils.Checkpoint

  defmodule TrainingClientStub do
    defstruct []

    def save_state(_client, name, _opts \\ []) do
      {:ok, Task.async(fn -> {:ok, %{path: "/tmp/#{name}-state"}} end)}
    end

    def save_weights_for_sampler(_client, name, _opts \\ []) do
      {:ok, Task.async(fn -> {:ok, %{path: "/tmp/#{name}-sampler"}} end)}
    end
  end

  setup do
    tmp_dir =
      Path.join(System.tmp_dir!(), "checkpoint_test_#{System.unique_integer([:positive])}")

    File.rm_rf!(tmp_dir)
    File.mkdir_p!(tmp_dir)
    %{tmp_dir: tmp_dir}
  end

  test "load_checkpoints_file returns empty when file missing", %{tmp_dir: tmp_dir} do
    assert Checkpoint.load_checkpoints_file(tmp_dir) == []
  end

  test "save_checkpoint writes checkpoints.jsonl", %{tmp_dir: tmp_dir} do
    client = %TrainingClientStub{}

    result =
      Checkpoint.save_checkpoint(
        client,
        "ckpt1",
        tmp_dir,
        %{"epoch" => 1, "batch" => 2},
        :both
      )

    assert result["state_path"] == "/tmp/ckpt1-state"
    assert result["sampler_path"] == "/tmp/ckpt1-sampler"

    [entry] = Checkpoint.load_checkpoints_file(tmp_dir)
    assert entry["name"] == "ckpt1"
    assert entry["epoch"] == 1
    assert entry["state_path"] == "/tmp/ckpt1-state"
  end

  test "get_last_checkpoint filters by required key", %{tmp_dir: tmp_dir} do
    client = %TrainingClientStub{}

    _ =
      Checkpoint.save_checkpoint(
        client,
        "ckpt1",
        tmp_dir,
        %{"epoch" => 1},
        :state
      )

    last = Checkpoint.get_last_checkpoint(tmp_dir, "state_path")
    assert last["name"] == "ckpt1"
  end
end
