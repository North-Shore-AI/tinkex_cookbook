defmodule TinkexCookbook.Parity.RecipesParityTest do
  use ExUnit.Case, async: true

  alias CrucibleTrain.Renderers.TrainOnWhat
  alias TinkexCookbook.Recipes.{ChatSl, Distillation, DPO, MathRl, SlBasic}

  test "sl_basic defaults align with python recipe" do
    config = SlBasic.default_config()

    assert config.model == "meta-llama/Llama-3.1-8B"
    assert config.learning_rate == 2.0e-4
    assert config.batch_size == 128
    assert config.max_length == 32_768
    assert config.train_on == TrainOnWhat.all_assistant_messages()
  end

  test "chat_sl defaults align with python CLI" do
    config = ChatSl.default_config()

    assert config.model == "meta-llama/Llama-3.1-8B"
    assert config.dataset == :no_robots
    assert config.learning_rate == 1.0e-4
    assert config.batch_size == 256
    assert config.max_length == 16_384
  end

  test "dpo defaults align with python CLI" do
    config = DPO.default_config()

    assert config.model == "meta-llama/Llama-3.2-1B"
    assert config.dataset == :hhh
    assert config.dpo_beta == 0.1
    assert config.batch_size == 256
  end

  test "math_rl defaults use gsm8k focus env" do
    config = MathRl.default_config()

    assert config.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert config.dataset == :gsm8k
    assert config.group_size == 4
  end

  test "distillation defaults include teacher/student models" do
    config = Distillation.default_config()

    assert config.dataset == :deepmath
    assert is_binary(config.model)
    assert is_binary(config.teacher_model)
  end
end
