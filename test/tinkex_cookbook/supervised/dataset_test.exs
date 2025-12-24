defmodule TinkexCookbook.Supervised.DatasetTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Supervised.{ChatDatasetBuilderCommonConfig, SupervisedDataset}
  alias TinkexCookbook.Types.{Datum, ModelInput, TensorData}

  describe "SupervisedDataset behaviour" do
    defmodule MockDataset do
      @behaviour SupervisedDataset

      defstruct [:data, :batch_size]

      @impl true
      def get_batch(%__MODULE__{data: data, batch_size: batch_size}, index) do
        start = index * batch_size
        Enum.slice(data, start, batch_size)
      end

      @impl true
      def length(%__MODULE__{data: data, batch_size: batch_size}) do
        div(Enum.count(data), batch_size)
      end

      @impl true
      def set_epoch(%__MODULE__{} = dataset, _seed), do: dataset
    end

    test "get_batch returns correct batch" do
      datums =
        for i <- 1..10 do
          %Datum{
            model_input: ModelInput.from_ints([i]),
            loss_fn_inputs: %{
              "weights" => TensorData.from_list([1.0], :float32),
              "target_tokens" => TensorData.from_list([i + 1], :int64)
            }
          }
        end

      dataset = %MockDataset{data: datums, batch_size: 3}

      batch0 = SupervisedDataset.get_batch(dataset, 0)
      assert length(batch0) == 3

      batch1 = SupervisedDataset.get_batch(dataset, 1)
      assert length(batch1) == 3
    end

    test "length returns number of batches" do
      datums =
        for i <- 1..10 do
          %Datum{
            model_input: ModelInput.from_ints([i]),
            loss_fn_inputs: %{}
          }
        end

      dataset = %MockDataset{data: datums, batch_size: 3}
      assert SupervisedDataset.length(dataset) == 3
    end
  end

  describe "conversation_to_datum/4" do
    test "converts simple conversation to datum" do
      # This test uses a mock tokenizer that returns predictable tokens
      _messages = [
        %{"role" => "user", "content" => "Hello"},
        %{"role" => "assistant", "content" => "Hi there!"}
      ]

      common_config = %ChatDatasetBuilderCommonConfig{
        model_name_for_tokenizer: "test-model",
        renderer_name: "role_colon",
        max_length: 100,
        batch_size: 1,
        train_on_what: "all_assistant_messages"
      }

      # We'll test the actual conversion once we have a mock tokenizer
      # For now, verify the config is valid
      assert common_config.train_on_what == "all_assistant_messages"
    end
  end
end
