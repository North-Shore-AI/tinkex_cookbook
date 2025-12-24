defmodule TinkexCookbook.Supervised.TrainTest do
  @moduledoc """
  Tests for the supervised training module.

  Uses mock Tinkex clients to verify training loop behavior
  without making network calls.
  """
  use ExUnit.Case, async: true

  alias TinkexCookbook.Supervised.{SupervisedDatasetFromList, Train}
  alias TinkexCookbook.Test.MockTinkex
  alias TinkexCookbook.Types.{Datum, EncodedTextChunk, ModelInput, TensorData}

  # Helper to create test datums
  defp create_test_datum do
    # Simple datum with 10 tokens
    tokens = Enum.to_list(1..10)
    weights = List.duplicate(1.0, 10)

    model_input = ModelInput.new([EncodedTextChunk.new(tokens)])
    weights_tensor = TensorData.from_list(weights, :float32)

    Datum.new(model_input, %{"weights" => weights_tensor})
  end

  describe "batch_datums/2" do
    test "batches datums into groups of specified size" do
      datums = for _ <- 1..5, do: create_test_datum()

      batches = Train.batch_datums(datums, 2)

      assert length(batches) == 2
      assert length(Enum.at(batches, 0)) == 2
      assert length(Enum.at(batches, 1)) == 2
      # Note: 5th datum is dropped because it doesn't fill a batch
    end

    test "returns empty list for empty input" do
      batches = Train.batch_datums([], 2)
      assert batches == []
    end

    test "handles exact batch size multiple" do
      datums = for _ <- 1..4, do: create_test_datum()

      batches = Train.batch_datums(datums, 2)

      assert length(batches) == 2
    end
  end

  describe "training_step/3" do
    test "executes forward_backward and optim_step" do
      training_client = MockTinkex.TrainingClient.new()
      batch = [create_test_datum()]
      adam_params = %{learning_rate: 1.0e-4, beta1: 0.9, beta2: 0.95, eps: 1.0e-8}

      result = Train.training_step(training_client, batch, adam_params)

      assert {:ok, metrics} = result
      assert is_map(metrics)
      assert Map.has_key?(metrics, "loss")
    end

    test "returns loss in metrics" do
      training_client = MockTinkex.TrainingClient.new()
      batch = [create_test_datum()]
      adam_params = %{learning_rate: 1.0e-4, beta1: 0.9, beta2: 0.95, eps: 1.0e-8}

      {:ok, metrics} = Train.training_step(training_client, batch, adam_params)

      assert metrics["loss"] > 0
    end
  end

  describe "compute_lr/3" do
    test "linear schedule decreases learning rate" do
      base_lr = 1.0e-4
      total_steps = 100

      lr_start = Train.compute_lr(base_lr, 0, total_steps, :linear)
      lr_mid = Train.compute_lr(base_lr, 50, total_steps, :linear)
      lr_end = Train.compute_lr(base_lr, 99, total_steps, :linear)

      assert lr_start > lr_mid
      assert lr_mid > lr_end
    end

    test "constant schedule keeps learning rate fixed" do
      base_lr = 1.0e-4
      total_steps = 100

      lr_start = Train.compute_lr(base_lr, 0, total_steps, :constant)
      lr_mid = Train.compute_lr(base_lr, 50, total_steps, :constant)
      lr_end = Train.compute_lr(base_lr, 99, total_steps, :constant)

      assert lr_start == base_lr
      assert lr_mid == base_lr
      assert lr_end == base_lr
    end

    test "cosine schedule decreases learning rate" do
      base_lr = 1.0e-4
      total_steps = 100

      lr_start = Train.compute_lr(base_lr, 0, total_steps, :cosine)
      lr_mid = Train.compute_lr(base_lr, 50, total_steps, :cosine)
      lr_end = Train.compute_lr(base_lr, 99, total_steps, :cosine)

      assert lr_start > lr_mid
      assert lr_mid > lr_end
    end
  end

  describe "run_epoch/4" do
    setup do
      training_client = MockTinkex.TrainingClient.new()
      datums = for _ <- 1..4, do: create_test_datum()
      dataset = SupervisedDatasetFromList.new(datums, 2)

      config = %{
        learning_rate: 1.0e-4,
        lr_schedule: :linear,
        adam_beta1: 0.9,
        adam_beta2: 0.95,
        adam_eps: 1.0e-8,
        total_steps: 2
      }

      %{training_client: training_client, dataset: dataset, config: config}
    end

    test "processes all batches in dataset", context do
      {:ok, metrics_list} =
        Train.run_epoch(
          context.training_client,
          context.dataset,
          0,
          context.config
        )

      # Dataset has 2 batches
      assert length(metrics_list) == 2
    end

    test "returns metrics for each step", context do
      {:ok, metrics_list} =
        Train.run_epoch(
          context.training_client,
          context.dataset,
          0,
          context.config
        )

      Enum.each(metrics_list, fn metrics ->
        assert Map.has_key?(metrics, "loss")
        assert Map.has_key?(metrics, "step")
      end)
    end
  end

  describe "TrainConfig struct" do
    test "can be created with defaults" do
      config =
        Train.TrainConfig.new(
          model_name: "test-model",
          log_path: "/tmp/test"
        )

      assert config.model_name == "test-model"
      assert config.learning_rate == 1.0e-4
      assert config.num_epochs == 1
    end

    test "accepts overrides" do
      config =
        Train.TrainConfig.new(
          model_name: "test-model",
          log_path: "/tmp/test",
          learning_rate: 2.0e-4,
          num_epochs: 3
        )

      assert config.learning_rate == 2.0e-4
      assert config.num_epochs == 3
    end
  end
end
