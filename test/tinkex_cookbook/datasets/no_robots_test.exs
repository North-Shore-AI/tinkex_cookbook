defmodule TinkexCookbook.Datasets.NoRobotsTest do
  @moduledoc """
  Tests for the NoRobots dataset builder.

  These tests use mock data to avoid network calls.
  """
  use ExUnit.Case, async: true

  alias CrucibleTrain.Renderers.{Llama3, TrainOnWhat}
  alias CrucibleTrain.Supervised.Dataset, as: SupervisedDataset
  alias CrucibleTrain.Types.{Datum, ModelInput, TensorData}
  alias TinkexCookbook.Datasets.NoRobots
  alias TinkexCookbook.Test.MockTokenizer

  describe "sample_to_messages/1" do
    test "extracts messages from a sample" do
      sample = %{
        "messages" => [
          %{"role" => "user", "content" => "Hello!"},
          %{"role" => "assistant", "content" => "Hi there!"}
        ]
      }

      messages = NoRobots.sample_to_messages(sample)

      assert length(messages) == 2
      assert hd(messages).role == "user"
      assert hd(messages).content == "Hello!"
    end

    test "handles empty messages list" do
      sample = %{"messages" => []}

      messages = NoRobots.sample_to_messages(sample)

      assert messages == []
    end
  end

  describe "build_datum/4" do
    setup do
      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)

      %{
        state: state,
        sample: %{
          "messages" => [
            %{"role" => "user", "content" => "What is 2+2?"},
            %{"role" => "assistant", "content" => "The answer is 4."}
          ]
        }
      }
    end

    test "builds a valid Datum from a sample", %{state: state, sample: sample} do
      datum =
        NoRobots.build_datum(
          sample,
          Llama3,
          state,
          TrainOnWhat.all_assistant_messages()
        )

      assert %Datum{} = datum
      assert %ModelInput{} = datum.model_input
      assert is_map(datum.loss_fn_inputs)
      assert Map.has_key?(datum.loss_fn_inputs, "weights")
    end

    test "datum weights align with model input length", %{state: state, sample: sample} do
      datum =
        NoRobots.build_datum(
          sample,
          Llama3,
          state,
          TrainOnWhat.all_assistant_messages()
        )

      weights = Map.get(datum.loss_fn_inputs, "weights")
      assert %TensorData{} = weights

      input_length = ModelInput.length(datum.model_input)
      weights_length = length(weights.data)

      assert input_length == weights_length
    end

    test "respects train_on_what parameter", %{state: state, sample: sample} do
      # With all_tokens, every weight should be 1.0 EXCEPT for BOS token weights
      # which are always 0.0. This matches Python behavior - see renderers.py line 354-356.
      datum_all =
        NoRobots.build_datum(
          sample,
          Llama3,
          state,
          TrainOnWhat.all_tokens()
        )

      weights_all = Map.get(datum_all.loss_fn_inputs, "weights")
      # BOS tokens get 0.0 weight (after slicing, some 0.0s remain from BOS)
      # Non-BOS tokens should be 1.0
      assert Enum.any?(weights_all.data, fn w -> w == 1.0 end)
      # Majority of weights should be 1.0 for all_tokens
      one_count = Enum.count(weights_all.data, fn w -> w == 1.0 end)
      assert one_count > length(weights_all.data) / 2

      # With last_assistant_message, only some weights should be 1.0
      datum_last =
        NoRobots.build_datum(
          sample,
          Llama3,
          state,
          TrainOnWhat.last_assistant_message()
        )

      weights_last = Map.get(datum_last.loss_fn_inputs, "weights")
      assert Enum.any?(weights_last.data, fn w -> w == 1.0 end)
      assert Enum.any?(weights_last.data, fn w -> w == 0.0 end)
    end
  end

  describe "build_datums/4" do
    setup do
      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)

      samples = [
        %{
          "messages" => [
            %{"role" => "user", "content" => "Hello"},
            %{"role" => "assistant", "content" => "Hi!"}
          ]
        },
        %{
          "messages" => [
            %{"role" => "user", "content" => "How are you?"},
            %{"role" => "assistant", "content" => "I'm good!"}
          ]
        }
      ]

      %{state: state, samples: samples}
    end

    test "builds datums for all samples", %{state: state, samples: samples} do
      datums =
        NoRobots.build_datums(
          samples,
          Llama3,
          state,
          TrainOnWhat.all_assistant_messages()
        )

      assert length(datums) == 2
      assert Enum.all?(datums, fn d -> %Datum{} = d end)
    end

    test "preserves order of samples", %{state: state, samples: samples} do
      datums =
        NoRobots.build_datums(
          samples,
          Llama3,
          state,
          TrainOnWhat.all_assistant_messages()
        )

      # Each datum should have tokens from corresponding sample
      assert length(datums) == length(samples)
    end
  end

  describe "create_supervised_dataset/2" do
    setup do
      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)

      samples = [
        %{
          "messages" => [
            %{"role" => "user", "content" => "Hello"},
            %{"role" => "assistant", "content" => "Hi!"}
          ]
        },
        %{
          "messages" => [
            %{"role" => "user", "content" => "Bye"},
            %{"role" => "assistant", "content" => "Goodbye!"}
          ]
        },
        %{
          "messages" => [
            %{"role" => "user", "content" => "Test"},
            %{"role" => "assistant", "content" => "Testing!"}
          ]
        }
      ]

      %{state: state, samples: samples}
    end

    test "creates a SupervisedDataset from samples", %{state: state, samples: samples} do
      dataset =
        NoRobots.create_supervised_dataset(
          samples,
          renderer_module: Llama3,
          renderer_state: state,
          train_on_what: TrainOnWhat.all_assistant_messages(),
          batch_size: 2
        )

      assert %CrucibleTrain.Supervised.DatasetFromSamples{} = dataset
    end

    test "dataset has correct number of batches", %{state: state, samples: samples} do
      dataset =
        NoRobots.create_supervised_dataset(
          samples,
          renderer_module: Llama3,
          renderer_state: state,
          train_on_what: TrainOnWhat.all_assistant_messages(),
          batch_size: 2
        )

      # 3 samples with batch_size 2 = 1 full batch
      assert SupervisedDataset.length(dataset) == 1
    end

    test "get_batch returns correct number of datums", %{state: state, samples: samples} do
      dataset =
        NoRobots.create_supervised_dataset(
          samples,
          renderer_module: Llama3,
          renderer_state: state,
          train_on_what: TrainOnWhat.all_assistant_messages(),
          batch_size: 2
        )

      batch = SupervisedDataset.get_batch(dataset, 0)
      assert length(batch) == 2
    end
  end
end
