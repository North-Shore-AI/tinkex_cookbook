defmodule TinkexCookbook.MixProject do
  use Mix.Project

  @version "0.3.1"
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
      extras: ["README.md", "CHANGELOG.md"]
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
      # HuggingFace ecosystem (maps to Python datasets + huggingface_hub)
      {:hf_datasets_ex, "~> 0.1.1"},
      {:hf_hub, "~> 0.1"},

      # Google Gemini API (maps to Python google-genai)
      {:gemini_ex, "~> 0.8", only: :prod},

      # OpenAI API (maps to Python openai)
      {:openai_ex, "~> 0.8", only: :prod},

      # LLM agent SDKs (CLI-backed)
      {:claude_agent_sdk, "~> 0.6.8"},
      {:codex_sdk, "~> 0.4.2"},

      # ChromaDB vector store (maps to Python chromadb)
      {:chroma, "~> 0.1.2"},

      # Configuration + CLI parsing (maps to Python chz)
      {:chz_ex, "~> 0.1.2"},

      # Schema validation (maps to Python pydantic)
      {:sinter, "~> 0.0.1"},

      # Python bridge for libs like sympy, pylatexenc, math_verify
      {:snakebridge, "~> 0.3.0"},

      # Tinkex client for training API (maps to Python tinker)
      {:tinkex, "~> 0.3.2"},

      # North-Shore-AI ecosystem (inspect-ai parity for evaluation)
      {:crucible_harness, "~> 0.3.2"},
      {:crucible_bench, "~> 0.3.1"},
      {:eval_ex, "~> 0.1.2"},
      {:crucible_datasets, "~> 0.5.1"},

      # Nx for tensor operations (replaces numpy/torch tensor ops)
      {:nx, "~> 0.9"},

      # Terminal tables (replaces Python rich)
      {:table_rex, "~> 4.0"},

      # S3/cloud storage (replaces Python blobfile for cloud paths)
      {:ex_aws, "~> 2.5"},
      {:ex_aws_s3, "~> 2.5"},

      # Development/test dependencies
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:mox, "~> 1.0", only: :test}
    ]
  end
end
