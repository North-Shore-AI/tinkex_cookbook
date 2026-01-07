defmodule TinkexCookbook.Recipes.DistillationTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Recipes.Distillation

  describe "recipe metadata" do
    test "name and description" do
      assert Distillation.name() == :distillation
      assert is_binary(Distillation.description())
    end

    test "uses distillation workflow" do
      assert Distillation.workflow() == CrucibleKitchen.Workflows.Distillation.__workflow__()
    end

    test "requires training, dataset, and sampling adapters" do
      required = Distillation.required_adapters()
      assert :training_client in required
      assert :dataset_store in required
      assert :sampling_client in required
    end
  end

  describe "defaults" do
    test "default_config includes teacher and student models" do
      config = Distillation.default_config()

      assert is_binary(config.model)
      assert is_binary(config.teacher_model)
      assert config.dataset == :deepmath
      assert config.temperature == 1.0
    end
  end
end
