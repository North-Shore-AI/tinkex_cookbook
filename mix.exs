defmodule TinkexCookbook.MixProject do
  use Mix.Project

  @version "0.4.0"
  @source_url "https://github.com/North-Shore-AI/tinkex_cookbook"

  def project do
    [
      app: :tinkex_cookbook,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: description(),
      package: package(),
      docs: docs(),
      dialyzer: dialyzer()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp description do
    """
    Elixir port of tinker-cookbook: training and evaluation recipes for the Tinker ML platform.
    """
  end

  defp package do
    [
      name: "tinkex_cookbook",
      files: ~w(lib .formatter.exs mix.exs README.md CHANGELOG.md LICENSE),
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url}
    ]
  end

  defp docs do
    [
      main: "readme",
      source_url: @source_url,
      extras: [
        "README.md",
        "CHANGELOG.md",
        "docs/guides/internal_architecture.md"
      ],
      groups_for_extras: [
        Guides: ~r/docs\/guides\/.*/
      ]
    ]
  end

  defp dialyzer do
    [
      plt_add_apps: [:mix, :ex_unit],
      plt_file: {:no_warn, "priv/plts/dialyzer.plt"}
    ]
  end

  defp deps do
    [
      # ==========================================================================
      # CORE ORCHESTRATION
      # crucible_kitchen provides ALL crucible_* libs, tinkex, hf_*, eval_ex,
      # chz_ex, telemetry, etc. as transitive dependencies
      # ==========================================================================
      {:crucible_kitchen, path: "../crucible_kitchen", override: true},

      # ==========================================================================
      # DIRECT DEPENDENCIES (not provided by crucible_kitchen)
      # ==========================================================================

      # LLM agent SDKs (CLI-backed) - used directly in adapters
      {:claude_agent_sdk, "~> 0.6.8"},
      {:codex_sdk, "~> 0.4.2"},

      # ChromaDB vector store - used directly in adapters/vector_store/chroma.ex
      {:chroma, "~> 0.1.2"},

      # ==========================================================================
      # DEVELOPMENT/TEST DEPENDENCIES
      # ==========================================================================
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:mox, "~> 1.0", only: :test}
    ]
  end
end
