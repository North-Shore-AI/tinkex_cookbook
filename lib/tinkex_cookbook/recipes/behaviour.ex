defmodule TinkexCookbook.Recipe do
  @moduledoc """
  Behaviour for TinkexCookbook recipes.

  Recipes are orchestration units that define:
  - A name and description
  - A configuration schema (ChzEx)
  - How to build a CrucibleIR.Experiment from config

  ## Example

      defmodule MyRecipe do
        @behaviour TinkexCookbook.Recipe

        @impl true
        def name, do: "my_recipe"

        @impl true
        def description, do: "My custom recipe"

        @impl true
        def config_schema, do: MyRecipe.Config

        @impl true
        def build_spec(config) do
          %CrucibleIR.Experiment{
            name: "my_recipe",
            stages: [...]
          }
        end
      end
  """

  @doc "Unique recipe name"
  @callback name() :: String.t()

  @doc "Human-readable description"
  @callback description() :: String.t()

  @doc "ChzEx config schema module"
  @callback config_schema() :: module()

  @doc "Build experiment spec from config"
  @callback build_spec(config :: struct() | map()) :: CrucibleIR.Experiment.t()

  @doc "Default configuration values"
  @callback default_config() :: map()

  @optional_callbacks [default_config: 0]
end
