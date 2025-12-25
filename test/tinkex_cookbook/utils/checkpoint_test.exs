defmodule TinkexCookbook.Utils.CheckpointTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureLog

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
    log =
      capture_log(fn ->
        assert Checkpoint.load_checkpoints_file(tmp_dir) == []
      end)

    assert log =~ "No checkpoints found at"
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

    log =
      capture_log(fn ->
        [entry] = Checkpoint.load_checkpoints_file(tmp_dir)
        assert entry["name"] == "ckpt1"
        assert entry["epoch"] == 1
        assert entry["state_path"] == "/tmp/ckpt1-state"
      end)

    assert log =~ "Reading checkpoints from"
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

    log =
      capture_log(fn ->
        last = Checkpoint.get_last_checkpoint(tmp_dir, "state_path")
        assert last["name"] == "ckpt1"
      end)

    assert log =~ "Found 1 valid checkpoints"
    assert log =~ "Using last checkpoint"
  end

  describe "checkpoint resume flow" do
    test "can resume training from last checkpoint", %{tmp_dir: tmp_dir} do
      client = %TrainingClientStub{}

      # Simulate first run - save checkpoint at step 5
      Checkpoint.save_checkpoint(
        client,
        "000005",
        tmp_dir,
        %{"epoch" => 0, "batch" => 5},
        :state
      )

      # Verify checkpoint was saved
      assert File.exists?(Path.join(tmp_dir, "checkpoints.jsonl"))

      # Simulate resume - find last checkpoint
      log =
        capture_log(fn ->
          last = Checkpoint.get_last_checkpoint(tmp_dir, "state_path")
          assert last != nil
          assert last["name"] == "000005"
          assert last["epoch"] == 0
          assert last["batch"] == 5

          # Resume from next batch
          resume_batch = last["batch"] + 1
          assert resume_batch == 6
        end)

      assert log =~ "Using last checkpoint"
    end

    test "multiple checkpoints returns the last one", %{tmp_dir: tmp_dir} do
      client = %TrainingClientStub{}

      # Save multiple checkpoints
      Checkpoint.save_checkpoint(client, "000005", tmp_dir, %{"epoch" => 0, "batch" => 5}, :state)

      Checkpoint.save_checkpoint(
        client,
        "000010",
        tmp_dir,
        %{"epoch" => 0, "batch" => 10},
        :state
      )

      Checkpoint.save_checkpoint(
        client,
        "000015",
        tmp_dir,
        %{"epoch" => 0, "batch" => 15},
        :state
      )

      log =
        capture_log(fn ->
          last = Checkpoint.get_last_checkpoint(tmp_dir, "state_path")
          assert last["name"] == "000015"
          assert last["batch"] == 15
        end)

      assert log =~ "Found 3 valid checkpoints"
    end

    test "returns nil when no checkpoints with required key", %{tmp_dir: tmp_dir} do
      client = %TrainingClientStub{}

      # Save checkpoint with only sampler path
      Checkpoint.save_checkpoint(client, "000005", tmp_dir, %{"epoch" => 0}, :sampler)

      # Request state_path - should return nil
      log =
        capture_log(fn ->
          last = Checkpoint.get_last_checkpoint(tmp_dir, "state_path")
          assert last == nil
        end)

      assert log =~ "No checkpoints found with key state_path"
    end
  end
end
