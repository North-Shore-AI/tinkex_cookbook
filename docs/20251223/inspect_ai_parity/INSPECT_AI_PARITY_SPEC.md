# tinkex_cookbook: Thin Adapter Specification

**Date:** 2025-12-23
**Status:** Implementation Specification
**Purpose:** Define the THIN wiring layer for inspect-ai patterns

---

## Design Principle: Thin Adapter

tinkex_cookbook's eval layer is **just wiring code**. All reusable abstractions live in ecosystem libs:

| Abstraction | Lives In | NOT in tinkex_cookbook |
|-------------|----------|------------------------|
| Solver, TaskState, Generate | crucible_harness | ✓ |
| Task, Sample, Scorer | eval_ex | ✓ |
| MemoryDataset, Filter | crucible_datasets | ✓ |
| SamplingClient | tinkex (unchanged) | ✓ |

**tinkex_cookbook eval/ = ~100 LOC wiring only**

---

## What tinkex_cookbook Provides

### 1. TinkexGenerate Adapter

Implements `CrucibleHarness.Generate` using tinkex's SamplingClient.

**File:** `lib/tinkex_cookbook/eval/tinkex_generate.ex`

```elixir
defmodule TinkexCookbook.Eval.TinkexGenerate do
  @moduledoc """
  Adapter that implements CrucibleHarness.Generate using Tinkex.SamplingClient.
  This is the ONLY tinkex_cookbook-specific code for eval.
  """

  @behaviour CrucibleHarness.Generate

  defstruct [:client, :model]

  def new(opts \\ []) do
    %__MODULE__{
      client: Keyword.fetch!(opts, :client),
      model: Keyword.fetch!(opts, :model)
    }
  end

  @impl true
  def generate(%__MODULE__{client: client, model: model}, messages, config) do
    prompt = messages_to_prompt(messages)

    params = %{
      model: model,
      max_tokens: Map.get(config, :max_tokens, 1024),
      temperature: Map.get(config, :temperature, 0.0),
      stop: Map.get(config, :stop, [])
    }

    case Tinkex.SamplingClient.sample(client, prompt, params) do
      {:ok, response} ->
        {:ok, %{
          content: response.completion,
          finish_reason: response.stop_reason || "stop",
          usage: %{
            input_tokens: response.usage[:prompt_tokens] || 0,
            output_tokens: response.usage[:completion_tokens] || 0
          }
        }}
      {:error, reason} ->
        {:error, reason}
    end
  end

  defp messages_to_prompt(messages) do
    # Convert chat messages to Tinker prompt format
    messages
    |> Enum.map(fn
      %{role: "system", content: c} -> "[SYSTEM]\n#{c}"
      %{role: "user", content: c} -> "[USER]\n#{c}"
      %{role: "assistant", content: c} -> "[ASSISTANT]\n#{c}"
      %{"role" => r, "content" => c} -> "[#{String.upcase(r)}]\n#{c}"
    end)
    |> Enum.join("\n\n")
  end
end
```

**LOC:** ~45

### 2. Eval Runner (Wiring)

Wires up ecosystem components - no logic of its own.

**File:** `lib/tinkex_cookbook/eval/runner.ex`

```elixir
defmodule TinkexCookbook.Eval.Runner do
  @moduledoc """
  Wires up ecosystem components for evaluation.
  All logic lives in crucible_harness/eval_ex - this is just glue.
  """

  alias CrucibleHarness.{Solver, TaskState}
  alias EvalEx.{Task, Sample, Scorer}

  @doc """
  Run an evaluation task using tinkex as the LLM backend.

  ## Options
  - `:client` - Tinkex.SamplingClient (required)
  - `:model` - Model name string (required)
  - `:scorers` - List of scorer modules (optional, uses task default)
  """
  def run(task_module, opts) when is_atom(task_module) do
    # Build the generate adapter
    generate_adapter = TinkexCookbook.Eval.TinkexGenerate.new(
      client: Keyword.fetch!(opts, :client),
      model: Keyword.fetch!(opts, :model)
    )

    # Get task definition
    task = build_task(task_module)

    # Get samples
    samples = get_samples(task)

    # Create generate function for scorers
    generate_fn = fn messages, config ->
      CrucibleHarness.Generate.generate(generate_adapter, messages, config)
    end

    # Run evaluation on each sample
    results = Enum.map(samples, fn sample ->
      evaluate_sample(sample, task, generate_adapter, generate_fn, opts)
    end)

    aggregate_results(results)
  end

  defp build_task(task_module) do
    %{
      id: task_module.task_id(),
      name: task_module.name(),
      dataset: task_module.dataset(),
      scorers: task_module.scorers()
    }
  end

  defp get_samples(%{dataset: dataset}) when is_list(dataset), do: dataset
  defp get_samples(%{dataset: dataset_module}) when is_atom(dataset_module) do
    # Load from crucible_datasets if it's a dataset reference
    {:ok, dataset} = CrucibleDatasets.load(dataset_module)
    dataset.items
  end

  defp evaluate_sample(sample, task, generate_adapter, generate_fn, opts) do
    # 1. Create initial state from sample
    state = TaskState.new(sample)

    # 2. Run solver chain (just Generate for basic eval)
    solver = Solver.Generate.new(opts[:generate_config] || %{})

    generate_callback = fn state, config ->
      case CrucibleHarness.Generate.generate(generate_adapter, state.messages, config) do
        {:ok, response} ->
          {:ok, state
          |> TaskState.add_message(%{role: "assistant", content: response.content})
          |> TaskState.set_output(response)}
        error -> error
      end
    end

    case Solver.Generate.solve(solver, state, generate_callback) do
      {:ok, final_state} ->
        # 3. Score the output
        scored_sample = Sample.with_output(sample, final_state.output.content)
        scores = score_sample(scored_sample, task.scorers, generate_fn, opts)

        %{
          sample_id: sample.id,
          output: final_state.output,
          scores: scores,
          error: nil
        }
      {:error, reason} ->
        %{
          sample_id: sample.id,
          output: nil,
          scores: %{},
          error: EvalEx.Error.new(:other, inspect(reason), sample_id: sample.id)
        }
    end
  end

  defp score_sample(sample, scorers, generate_fn, opts) do
    Enum.reduce(scorers, %{}, fn scorer_module, acc ->
      scorer_opts = Keyword.merge(opts, [generate_fn: generate_fn])

      case scorer_module.score(sample, scorer_opts) do
        {:ok, score} -> Map.put(acc, scorer_module.scorer_id(), score)
        {:error, _} -> acc
      end
    end)
  end

  defp aggregate_results(results) do
    successful = Enum.reject(results, & &1.error)

    %{
      total: length(results),
      successful: length(successful),
      failed: length(results) - length(successful),
      results: results,
      aggregated_scores: aggregate_scores(successful)
    }
  end

  defp aggregate_scores(results) do
    results
    |> Enum.flat_map(fn r -> Map.to_list(r.scores) end)
    |> Enum.group_by(fn {k, _} -> k end, fn {_, v} -> v.value end)
    |> Enum.map(fn {scorer_id, values} ->
      {scorer_id, %{
        mean: Enum.sum(values) / length(values),
        count: length(values)
      }}
    end)
    |> Map.new()
  end
end
```

**LOC:** ~95

---

## File Structure

```
lib/tinkex_cookbook/eval/
├── tinkex_generate.ex    # CrucibleHarness.Generate adapter (~45 LOC)
└── runner.ex             # Wiring code (~55 LOC)
```

**Total tinkex_cookbook eval code:** ~100 LOC

---

## Dependencies

```elixir
# mix.exs
defp deps do
  [
    {:tinkex, path: "../tinkex"},
    {:crucible_harness, path: "../crucible_harness"},
    {:eval_ex, path: "../eval_ex"},
    {:crucible_datasets, path: "../crucible_datasets"}
  ]
end
```

---

## Usage Example

```elixir
# 1. Define a task using eval_ex abstractions
defmodule MyMathEval do
  use EvalEx.Task

  @impl true
  def task_id, do: "math_eval_v1"

  @impl true
  def name, do: "Math Problem Evaluation"

  @impl true
  def dataset do
    # Using crucible_datasets
    {:ok, ds} = CrucibleDatasets.load(:gsm8k, split: "test", limit: 100)
    Enum.map(ds.items, fn item ->
      EvalEx.Sample.new(
        input: item.input,
        target: item.expected,
        metadata: item.metadata
      )
    end)
  end

  @impl true
  def scorers, do: [EvalEx.Scorer.ExactMatch]
end

# 2. Run evaluation using tinkex_cookbook runner
{:ok, client} = Tinkex.SamplingClient.connect(api_key: "...")

results = TinkexCookbook.Eval.Runner.run(MyMathEval,
  client: client,
  model: "my-finetuned-model"
)

IO.puts "Accuracy: #{results.aggregated_scores["exact_match"].mean}"
```

---

## What This Replaces

The previous spec (~600 LOC in tinkex_cookbook) is replaced by:

| Old (tinkex_cookbook) | New Location | LOC |
|-----------------------|--------------|-----|
| Sample, Dataset types | eval_ex | ~40 |
| ModelAPI behaviour | crucible_harness (Generate) | ~20 |
| Task DSL | eval_ex (Task behaviour) | ~80 |
| Solvers | crucible_harness | ~55 |
| Scorers | eval_ex | ~70 |
| Runner logic | tinkex_cookbook (wiring) | ~55 |
| TinkexGenerate adapter | tinkex_cookbook | ~45 |

**Result:** 600 LOC → 100 LOC in tinkex_cookbook (the rest is reusable in ecosystem)

---

## Summary

| Component | LOC | Purpose |
|-----------|-----|---------|
| TinkexGenerate | 45 | Implements Generate using tinkex |
| Runner | 55 | Wires ecosystem components |
| **Total** | **~100** | **Just wiring** |

All reusable abstractions (Solver, Task, Sample, Scorer, Generate protocol) live in ecosystem libs and can be used by other projects without depending on tinkex_cookbook.

---

**Document Status:** Complete
**Last Updated:** 2025-12-23
