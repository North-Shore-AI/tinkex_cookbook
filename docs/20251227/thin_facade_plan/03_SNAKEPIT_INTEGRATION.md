# Snakepit Integration for Python Libraries

**Date:** 2025-12-27

This document details how Snakepit (the Elixir-Python gRPC bridge) integrates with TinkexCookbook recipes for accessing Python libraries like sympy, pylatexenc, and math_verify.

NOTE: Superseded by the corrected ownership model. Adapter implementations now
live in `crucible_kitchen`; TinkexCookbook provides recipes/config only.

---

## Integration Principle

**Snakepit calls happen in RECIPES, not in the FACADE core.**

The facade remains pure Elixir, delegating to crucible_train stages. Recipes that need Python capabilities call Snakepit directly within their recipe implementation or via specialized stages.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Recipe Layer                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              MathVerifyRecipe                               │ │
│  │  ┌──────────────┐      ┌───────────────────────────────┐  │ │
│  │  │ Training via │      │ Verification via Snakepit      │  │ │
│  │  │crucible_train│ ───► │  sympy │ pylatexenc │ math_v  │  │ │
│  │  └──────────────┘      └───────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Snakepit Pool                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Python gRPC Workers                                          ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    ││
│  │  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  │ Worker 4 │    ││
│  │  │  sympy   │  │  sympy   │  │  sympy   │  │  sympy   │    ││
│  │  │pylatexenc│  │pylatexenc│  │pylatexenc│  │pylatexenc│    ││
│  │  │math_verify│  │math_verify│  │math_verify│  │math_verify│    ││
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Required Python Libraries

Based on existing recipes and tinker-cookbook parity:

| Library | Version | Purpose | Used In |
|---------|---------|---------|---------|
| `sympy` | >= 1.12 | Symbolic mathematics | Math verification, equation solving |
| `pylatexenc` | >= 2.10 | LaTeX parsing/rendering | Math rendering, normalization |
| `math_verify` | >= 0.1.0 | Mathematical equivalence checking | GSM8K evaluation |
| `numpy` | >= 1.21 | Numerical operations | Tensor data transfer |
| `grpcio` | >= 1.60 | gRPC communication | Core bridge |
| `protobuf` | >= 4.25 | Serialization | Core bridge |

---

## Snakepit Adapter Setup

### 1. Create Python Adapter

Create a custom adapter for math operations:

```python
# priv/python/snakepit_bridge/adapters/math/math_adapter.py

from snakepit_bridge.base_adapter import BaseAdapter, tool
import sympy
from pylatexenc.latex2text import LatexNodes2Text
from math_verify import verify_math_equivalence

class MathAdapter(BaseAdapter):
    """Adapter for mathematical operations."""

    name = "math"
    description = "Mathematical operations via sympy, pylatexenc, math_verify"

    @tool
    def simplify_expression(self, expr: str) -> dict:
        """Simplify a mathematical expression using sympy."""
        try:
            parsed = sympy.sympify(expr)
            simplified = sympy.simplify(parsed)
            return {
                "success": True,
                "result": str(simplified),
                "latex": sympy.latex(simplified)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @tool
    def solve_equation(self, equation: str, variable: str = "x") -> dict:
        """Solve an equation for a variable."""
        try:
            var = sympy.Symbol(variable)
            eq = sympy.sympify(equation)
            solutions = sympy.solve(eq, var)
            return {
                "success": True,
                "solutions": [str(s) for s in solutions]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @tool
    def latex_to_text(self, latex: str) -> dict:
        """Convert LaTeX to plain text."""
        try:
            converter = LatexNodes2Text()
            text = converter.latex_to_text(latex)
            return {"success": True, "text": text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @tool
    def verify_math_answer(
        self,
        reference: str,
        prediction: str,
        tolerance: float = 1e-6
    ) -> dict:
        """Verify if two mathematical expressions are equivalent."""
        try:
            is_correct = verify_math_equivalence(
                reference,
                prediction,
                tolerance=tolerance
            )
            return {
                "success": True,
                "is_correct": is_correct,
                "reference": reference,
                "prediction": prediction
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @tool
    def batch_verify(
        self,
        pairs: list,  # List of {"reference": str, "prediction": str}
        tolerance: float = 1e-6
    ) -> dict:
        """Batch verify multiple math answer pairs."""
        results = []
        for pair in pairs:
            result = self.verify_math_answer(
                pair["reference"],
                pair["prediction"],
                tolerance
            )
            results.append(result)

        correct = sum(1 for r in results if r.get("is_correct", False))
        return {
            "success": True,
            "results": results,
            "accuracy": correct / len(results) if results else 0.0,
            "correct": correct,
            "total": len(results)
        }
```

### 2. Register Adapter

```python
# priv/python/snakepit_bridge/adapters/__init__.py

from .math.math_adapter import MathAdapter

ADAPTERS = [
    MathAdapter,
    # ... other adapters
]
```

### 3. Add Requirements

```txt
# priv/python/requirements.txt

# Core snakepit requirements
grpcio>=1.60.0
grpcio-tools>=1.60.0
protobuf>=4.25.0
orjson>=3.9.0
psutil>=5.9.0

# Math operations
sympy>=1.12
pylatexenc>=2.10
math_verify>=0.1.0

# Numerical operations
numpy>=1.21.0
```

---

## Elixir Integration Module

Create a convenience wrapper in tinkex_cookbook:

```elixir
defmodule TinkexCookbook.Python.Math do
  @moduledoc """
  Elixir interface for Python math operations via Snakepit.

  This module provides convenient wrappers around the Python
  math adapter's tools. Used by recipes that need mathematical
  verification (e.g., GSM8K evaluation).
  """

  @pool :math_pool

  @doc """
  Simplify a mathematical expression.

  ## Examples

      iex> Math.simplify("x^2 + 2*x + 1")
      {:ok, %{result: "(x + 1)^2", latex: "(x + 1)^{2}"}}
  """
  def simplify(expression) do
    case Snakepit.execute(@pool, "simplify_expression", %{expr: expression}) do
      {:ok, %{"success" => true} = result} ->
        {:ok, atomize_keys(result)}

      {:ok, %{"success" => false, "error" => error}} ->
        {:error, error}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Solve an equation for a variable.
  """
  def solve(equation, variable \\ "x") do
    case Snakepit.execute(@pool, "solve_equation", %{
      equation: equation,
      variable: variable
    }) do
      {:ok, %{"success" => true, "solutions" => solutions}} ->
        {:ok, solutions}

      {:ok, %{"success" => false, "error" => error}} ->
        {:error, error}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Verify if a predicted answer matches a reference answer.

  Used for evaluating math problem solutions (e.g., GSM8K).
  """
  def verify_answer(reference, prediction, tolerance \\ 1.0e-6) do
    case Snakepit.execute(@pool, "verify_math_answer", %{
      reference: reference,
      prediction: prediction,
      tolerance: tolerance
    }) do
      {:ok, %{"success" => true, "is_correct" => is_correct}} ->
        {:ok, is_correct}

      {:ok, %{"success" => false, "error" => error}} ->
        {:error, error}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Batch verify multiple answer pairs.

  More efficient than calling verify_answer/3 multiple times.
  """
  def batch_verify(pairs, tolerance \\ 1.0e-6) when is_list(pairs) do
    case Snakepit.execute(@pool, "batch_verify", %{
      pairs: pairs,
      tolerance: tolerance
    }) do
      {:ok, %{"success" => true} = result} ->
        {:ok, %{
          accuracy: result["accuracy"],
          correct: result["correct"],
          total: result["total"],
          results: Enum.map(result["results"], &atomize_keys/1)
        }}

      {:ok, %{"success" => false, "error" => error}} ->
        {:error, error}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Convert LaTeX to plain text.
  """
  def latex_to_text(latex) do
    case Snakepit.execute(@pool, "latex_to_text", %{latex: latex}) do
      {:ok, %{"success" => true, "text" => text}} ->
        {:ok, text}

      {:ok, %{"success" => false, "error" => error}} ->
        {:error, error}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp atomize_keys(map) when is_map(map) do
    Map.new(map, fn {k, v} ->
      key = if is_binary(k), do: String.to_atom(k), else: k
      {key, atomize_keys(v)}
    end)
  end
  defp atomize_keys(list) when is_list(list) do
    Enum.map(list, &atomize_keys/1)
  end
  defp atomize_keys(value), do: value
end
```

---

## Recipe Integration Example

### GSM8K Evaluation Recipe with Math Verification

```elixir
defmodule TinkexCookbook.Recipes.Gsm8kEval do
  @moduledoc """
  GSM8K evaluation recipe with mathematical verification.

  This recipe:
  1. Loads GSM8K test samples
  2. Generates solutions via trained model
  3. Verifies answers using Snakepit math adapter
  """

  @behaviour TinkexCookbook.Recipe

  alias CrucibleIR.{Experiment, StageDef}
  alias TinkexCookbook.Python.Math

  @impl true
  def name, do: "gsm8k_eval"

  @impl true
  def description, do: "Evaluate model on GSM8K with math verification"

  @impl true
  def config_schema, do: TinkexCookbook.Recipes.Gsm8kEvalConfig

  @impl true
  def build_spec(config) do
    %Experiment{
      name: "gsm8k_eval_#{config.model}",
      description: description(),
      stages: [
        # Stage 1: Load dataset
        %StageDef{
          name: "load_dataset",
          module: CrucibleDatasets.Stages.LoadDataset,
          options: %{
            dataset: "gsm8k",
            split: :test,
            limit: config.limit
          }
        },
        # Stage 2: Generate solutions
        %StageDef{
          name: "generate",
          module: TinkexCookbook.Eval.TinkexGenerate,
          options: %{
            model: config.model,
            temperature: config.temperature,
            max_tokens: config.max_tokens
          }
        },
        # Stage 3: Verify with Python math tools
        %StageDef{
          name: "verify",
          module: __MODULE__.VerifyStage,
          options: %{}
        }
      ]
    }
  end

  # Custom stage for math verification via Snakepit
  defmodule VerifyStage do
    @behaviour Crucible.Stage

    @impl true
    def call(context, _opts) do
      samples = context.assigns.samples
      generations = context.assigns.generations

      # Build pairs for batch verification
      pairs = Enum.zip_with(samples, generations, fn sample, gen ->
        %{
          reference: sample.answer,
          prediction: extract_answer(gen)
        }
      end)

      # Call Snakepit for batch math verification
      case TinkexCookbook.Python.Math.batch_verify(pairs) do
        {:ok, results} ->
          {:ok, Map.put(context.assigns, :verification, results)}

        {:error, reason} ->
          {:error, {:verification_failed, reason}}
      end
    end

    @impl true
    def describe do
      %{
        name: "gsm8k_verify",
        description: "Verify GSM8K answers using Python math libraries",
        inputs: [:samples, :generations],
        outputs: [:verification]
      }
    end

    defp extract_answer(generation) do
      # Extract numerical answer from generation
      # e.g., "The answer is 42" -> "42"
      case Regex.run(~r/(?:answer is|=)\s*([0-9.,]+)/, generation) do
        [_, answer] -> answer
        nil -> generation
      end
    end
  end
end
```

---

## Snakepit Pool Configuration

Configure Snakepit pools in your application:

```elixir
# config/config.exs

config :snakepit,
  pools: [
    # Pool for math operations
    math_pool: [
      size: 4,
      adapter: Snakepit.Adapters.GRPCPython,
      python_path: "priv/python",
      adapters: ["math"],
      lifecycle: [
        ttl: :timer.minutes(30),
        max_requests: 10_000
      ]
    ],
    # Pool for general Python operations (if needed)
    general_pool: [
      size: 2,
      adapter: Snakepit.Adapters.GRPCPython,
      python_path: "priv/python",
      adapters: ["showcase"],
      lifecycle: [
        ttl: :timer.hours(1)
      ]
    ]
  ]

config :snakepit, :hardware,
  detect_gpu: true,
  prefer_gpu: false  # Math operations don't need GPU
```

---

## Where Snakepit Goes

### In Core Facade: NO

The facade (`TinkexCookbook.Runtime`) does NOT call Snakepit directly:
- Facade handles port resolution and experiment execution
- Pure Elixir, no Python dependencies in the hot path

### In Recipes: YES

Recipes that need Python capabilities use Snakepit:
- Math verification recipes (GSM8K, MATH, etc.)
- Symbolic computation recipes
- Any recipe needing Python-only libraries

### In Custom Stages: YES

Custom stages can call Snakepit:
- `VerifyStage` for math verification
- `SymbolicComputeStage` for symbolic math
- Any stage needing Python libraries

### In Adapters: MAYBE

Some adapters might use Snakepit:
- `EmbeddingClient.SentenceTransformers` (if using Python embeddings)
- `VectorStore.FAISS` (if using FAISS via Python)

---

## Architecture Decision: Why Not in Facade?

1. **Separation of Concerns**: Facade handles orchestration, not computation
2. **Optional Dependency**: Not all recipes need Python
3. **Testability**: Facade can be tested without Python workers
4. **Performance**: No Python startup overhead for pure Elixir recipes
5. **Deployment Flexibility**: Can run subset of recipes without Python

```
┌────────────────────────────────────────────────────────────────┐
│                     TinkexCookbook                              │
│                                                                 │
│  ┌──────────────────────┐    ┌───────────────────────────────┐ │
│  │   Pure Elixir Path   │    │    Python-Enhanced Path       │ │
│  │                      │    │                               │ │
│  │  sl_basic            │    │  gsm8k_eval                   │ │
│  │  dpo                 │    │  math_reasoning               │ │
│  │  distill             │    │  symbolic_calc                │ │
│  │                      │    │                               │ │
│  │  No Snakepit needed  │    │  Uses Snakepit pools          │ │
│  └──────────────────────┘    └───────────────────────────────┘ │
│                                         │                       │
│                                         ▼                       │
│                               ┌─────────────────────┐          │
│                               │     Snakepit        │          │
│                               │   (Python Bridge)   │          │
│                               └─────────────────────┘          │
└────────────────────────────────────────────────────────────────┘
```

---

## Summary

| Component | Uses Snakepit? | Notes |
|-----------|----------------|-------|
| `TinkexCookbook.Runtime` | No | Pure Elixir facade |
| `TinkexCookbook.Recipes.SlBasic` | No | Pure Elixir training |
| `TinkexCookbook.Recipes.Gsm8kEval` | Yes | Math verification |
| `TinkexCookbook.Python.Math` | Yes | Convenience wrapper |
| `TinkexCookbook.Adapters.*` | Maybe | Only if needed |
| `crucible_train.*` | No | Pure Elixir |

Python libraries are accessed **only where needed**, keeping the core system lightweight and the Python bridge isolated to specific use cases.
