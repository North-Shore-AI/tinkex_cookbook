defmodule TinkexCookbook.Recipes.SlBasicConfigTest do
  use ExUnit.Case, async: true

  alias TinkexCookbook.Recipes.SlBasic.CliConfig

  test "entrypoint applies defaults with optional n_train_samples" do
    assert {:ok, %CliConfig{} = config} = ChzEx.entrypoint(CliConfig, [])
    assert config.n_train_samples == nil
  end
end
