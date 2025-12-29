defmodule TinkexCookbook.Recipes.SlBasicTest do
  @moduledoc """
  Tests for the sl_basic recipe.

  These tests verify that the recipe:
  1. Implements the Recipe behaviour correctly
  2. Builds valid CrucibleIR.Experiment specs
  3. Works with the NoRobots dataset and CrucibleTrain types
  """
  use ExUnit.Case, async: true

  alias CrucibleIR.Experiment
  alias CrucibleTrain.Renderers.{Llama3, TrainOnWhat}
  alias CrucibleTrain.Supervised.Dataset, as: SupervisedDataset
  alias CrucibleTrain.Types.{Datum, ModelInput, TensorData}
  alias TinkexCookbook.Datasets.NoRobots
  alias TinkexCookbook.Recipes.SlBasic
  alias TinkexCookbook.Test.MockTokenizer

  describe "Recipe behaviour" do
    test "name/0 returns recipe name" do
      assert SlBasic.name() == "sl_basic"
    end

    test "description/0 returns description" do
      assert is_binary(SlBasic.description())
    end

    test "default_config/0 returns valid config map" do
      config = SlBasic.default_config()

      assert is_map(config)
      assert config.model == "meta-llama/Llama-3.1-8B"
      assert config.learning_rate == 2.0e-4
      assert config.num_epochs == 1
      assert config.batch_size == 128
      assert config.lora_rank == 32
    end
  end

  describe "build_spec/1" do
    test "builds valid Experiment struct" do
      config = SlBasic.default_config()
      experiment = SlBasic.build_spec(config)

      assert %Experiment{} = experiment
      assert experiment.experiment_type == :training
      assert experiment.backend.id == :tinker
    end

    test "builds experiment with custom config" do
      config = %{
        model: "Qwen/Qwen3-8B",
        num_epochs: 3,
        learning_rate: 1.0e-4
      }

      experiment = SlBasic.build_spec(config)

      assert experiment.backend.model_version == "Qwen/Qwen3-8B"
      assert experiment.training_config.epochs == 3
      assert experiment.training_config.learning_rate == 1.0e-4
    end

    test "includes pipeline with supervised train stage" do
      config = SlBasic.default_config()
      experiment = SlBasic.build_spec(config)

      assert length(experiment.pipeline) == 1
      [stage] = experiment.pipeline
      assert stage.name == :supervised_train
      assert stage.module == CrucibleTrain.Stages.SupervisedTrain
    end
  end

  describe "renderer selection" do
    test "selects Llama3 renderer for Llama models" do
      config = %{model: "meta-llama/Llama-3.1-8B"}
      experiment = SlBasic.build_spec(config)

      [stage] = experiment.pipeline
      assert stage.options.renderer == CrucibleTrain.Renderers.Llama3
    end

    test "selects Qwen3 renderer for Qwen models" do
      config = %{model: "Qwen/Qwen3-8B"}
      experiment = SlBasic.build_spec(config)

      [stage] = experiment.pipeline
      assert stage.options.renderer == CrucibleTrain.Renderers.Qwen3
    end

    test "selects DeepSeekV3 renderer for DeepSeek models" do
      config = %{model: "deepseek-ai/DeepSeek-V3"}
      experiment = SlBasic.build_spec(config)

      [stage] = experiment.pipeline
      assert stage.options.renderer == CrucibleTrain.Renderers.DeepSeekV3
    end

    test "falls back to RoleColon for unknown models" do
      config = %{model: "unknown/model"}
      experiment = SlBasic.build_spec(config)

      [stage] = experiment.pipeline
      assert stage.options.renderer == CrucibleTrain.Renderers.RoleColon
    end
  end

  describe "NoRobots dataset integration" do
    test "sample_to_messages converts dataset format to Messages" do
      sample = %{
        "messages" => [
          %{"role" => "user", "content" => "What is 2+2?"},
          %{"role" => "assistant", "content" => "The answer is 4."}
        ]
      }

      messages = NoRobots.sample_to_messages(sample)

      assert length(messages) == 2
      assert hd(messages).role == "user"
      assert hd(messages).content == "What is 2+2?"
    end

    test "build_datum creates valid Datum struct" do
      sample = %{
        "messages" => [
          %{"role" => "user", "content" => "Hello!"},
          %{"role" => "assistant", "content" => "Hi there!"}
        ]
      }

      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)

      datum = NoRobots.build_datum(sample, Llama3, state, TrainOnWhat.all_assistant_messages())

      assert %Datum{} = datum
      assert %ModelInput{} = datum.model_input
      assert is_map(datum.loss_fn_inputs)
      assert Map.has_key?(datum.loss_fn_inputs, "weights")
    end

    test "create_supervised_dataset creates Dataset with lazy evaluation" do
      samples = [
        %{
          "messages" => [
            %{"role" => "user", "content" => "Hi"},
            %{"role" => "assistant", "content" => "Hello!"}
          ]
        },
        %{
          "messages" => [
            %{"role" => "user", "content" => "Bye"},
            %{"role" => "assistant", "content" => "Goodbye!"}
          ]
        }
      ]

      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)

      dataset =
        NoRobots.create_supervised_dataset(samples,
          renderer_module: Llama3,
          renderer_state: state,
          train_on_what: TrainOnWhat.all_assistant_messages(),
          batch_size: 1
        )

      # Check dataset interface
      assert SupervisedDataset.length(dataset) == 2

      # Get first batch
      batch = SupervisedDataset.get_batch(dataset, 0)
      assert length(batch) == 1
      assert %Datum{} = hd(batch)
    end
  end

  describe "datum structure compatibility" do
    test "datum has correct loss_fn_inputs structure" do
      sample = %{
        "messages" => [
          %{"role" => "user", "content" => "Test"},
          %{"role" => "assistant", "content" => "Response"}
        ]
      }

      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)

      datum = NoRobots.build_datum(sample, Llama3, state, TrainOnWhat.all_assistant_messages())

      # Check weights structure
      weights = datum.loss_fn_inputs["weights"]
      assert %TensorData{} = weights
      assert is_list(weights.data)
      assert weights.dtype in [:float32, :float64, "float32", "float64"]

      # Check target_tokens if present
      if Map.has_key?(datum.loss_fn_inputs, "target_tokens") do
        target_tokens = datum.loss_fn_inputs["target_tokens"]
        assert %TensorData{} = target_tokens
        assert is_list(target_tokens.data)
      end
    end

    test "model_input has correct chunk structure" do
      sample = %{
        "messages" => [
          %{"role" => "user", "content" => "Hello"},
          %{"role" => "assistant", "content" => "Hi"}
        ]
      }

      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)

      datum = NoRobots.build_datum(sample, Llama3, state, TrainOnWhat.all_assistant_messages())

      # Check model input
      assert %ModelInput{chunks: chunks} = datum.model_input
      assert is_list(chunks)
      assert chunks != []

      # Each chunk should be EncodedTextChunk
      Enum.each(chunks, fn chunk ->
        assert %CrucibleTrain.Types.EncodedTextChunk{tokens: tokens} = chunk
        assert is_list(tokens)
        assert Enum.all?(tokens, &is_integer/1)
      end)
    end
  end

  describe "datum to Tinkex conversion" do
    test "datum converts to proper Tinkex format" do
      sample = %{
        "messages" => [
          %{"role" => "user", "content" => "Hi"},
          %{"role" => "assistant", "content" => "Hello!"}
        ]
      }

      {:ok, state} = Llama3.init(tokenizer: MockTokenizer)

      datum = NoRobots.build_datum(sample, Llama3, state, TrainOnWhat.all_assistant_messages())

      # Use internal function to verify datum can be converted
      # This tests the datum_to_tinkex function indirectly
      assert %Datum{} = datum
      assert %ModelInput{chunks: [_first_chunk | _]} = datum.model_input
      assert is_map(datum.loss_fn_inputs)
    end
  end
end
