# Elixir Mapping and Gap Analysis (Cookbook Eval Path)

This mapping focuses on the inspect-ai features used by tinker-cookbook and
how they map to `tinkex_cookbook` + `crucible_*` + `eval_ex`.

## Direct Mapping Table

Python usage (file:line) | inspect-ai API | Elixir mapping (file:line) | Parity notes
---|---|---|---
`tinker-cookbook/tinker_cookbook/eval/inspect_utils.py:57-156` | `ModelAPI.generate(...)` + `ModelOutput` | `lib/tinkex_cookbook/eval/tinkex_generate.ex:61-179` + `../crucible_harness/lib/research_harness/generate.ex:1-172` | Partial: response shape OK; usage counts are approximated and tools are ignored.
`tinker-cookbook/tinker_cookbook/eval/inspect_evaluators.py:76-106` | `eval_async(tasks, model, ...)` | `lib/tinkex_cookbook/eval/runner.ex:76-266` | Partial: no async eval runner, log_dir handling, retries, or limits.
`tinker-cookbook/tinker_cookbook/eval/custom_inspect_task.py:50-68` | `@task` + `Task` | `../eval_ex/lib/eval_ex/task.ex:1-123` + `../eval_ex/lib/eval_ex/task/registry.ex:1-93` | Partial: EvalEx has behaviour + registry, no decorator macro or task loader.
`tinker-cookbook/tinker_cookbook/eval/custom_inspect_task.py:21-33` | `MemoryDataset` + `Sample` | `../crucible_datasets/lib/dataset_manager/memory_dataset.ex:1-105` + `../eval_ex/lib/eval_ex/sample.ex:1-93` | Partial: Dataset and Sample are split across two libs; no built-in bridge.
`tinker-cookbook/tinker_cookbook/eval/custom_inspect_task.py:61` | `solver.generate()` | `../crucible_harness/lib/research_harness/solver/generate.ex:1-182` | Partial: no tool-call loop equivalent to inspect-ai.
`tinker-cookbook/tinker_cookbook/eval/custom_inspect_task.py:62-68` | `model_graded_qa()` | `../eval_ex/lib/eval_ex/scorer/llm_judge.ex:1-84` | Partial: LLMJudge parses CORRECT/INCORRECT; inspect-ai expects GRADE: C/I/P patterns and optional partial credit.
`tinker-cookbook/tinker_cookbook/eval/inspect_utils.py:100-106` | `tools` + `tool_choice` params | None | Missing: tool calling types and wiring in Elixir.
`tinker-cookbook/tinker_cookbook/recipes/chat_sl/train.py:88-108` | inline `inspect_evals/*` tasks | None | Missing: task registry/loader for inspect_evals tasks.

## Elixir Eval Stack Used Today

- `TinkexCookbook.Eval.TinkexGenerate` bridges `SamplingClient` to the eval
  runner: `lib/tinkex_cookbook/eval/tinkex_generate.ex:61-179`.
- `TinkexCookbook.Eval.Runner` is a synchronous evaluator:
  `lib/tinkex_cookbook/eval/runner.ex:76-266`.
- EvalEx models tasks and samples:
  - `../eval_ex/lib/eval_ex/task.ex:1-123`
  - `../eval_ex/lib/eval_ex/sample.ex:1-93`
- CrucibleHarness provides solver + generate behaviour:
  - `../crucible_harness/lib/research_harness/solver.ex:1-109`
  - `../crucible_harness/lib/research_harness/solver/generate.ex:1-182`
  - `../crucible_harness/lib/research_harness/task_state.ex:1-160`

## Gaps vs inspect-ai (Cookbook Usage Only)

1. Task loading and registry integration
   - inspect-ai supports string task specs and loader resolution.
   - EvalEx has a registry but no loader for `inspect_evals/*`.

2. Async evaluation runner
   - inspect-ai `eval_async` manages concurrency, limits, retries, and logs.
   - `TinkexCookbook.Eval.Runner` is sequential and does not log to files.

3. Tool calling surface
   - inspect-ai `ModelAPI.generate` includes `tools` + `tool_choice`.
   - No tool types exist in EvalEx/CrucibleHarness; `TinkexGenerate` ignores them.

4. LLM-as-judge parity
   - inspect-ai `model_graded_qa` supports grade regex, partial credit, and grader roles.
   - `EvalEx.Scorer.LLMJudge` is a minimal CORRECT/INCORRECT parser.

5. Usage accounting
   - inspect-ai `ModelUsage` counts prompt + completion tokens.
   - `TinkexGenerate` currently uses string length and sets `prompt_tokens` to 0.

6. Multi-part message content
   - inspect-ai accepts `Content` lists with images, reasoning, and tool calls.
   - Cookbook adapter only accepts `str` content (`inspect_utils.py:45-55`).

## Minimum Parity Needed for Current Cookbook Recipes

To reproduce the cookbook behavior (inline evals + custom task example), the
Elixir stack must provide:

1. A task abstraction with datasets of `Sample` records.
2. A generate adapter that builds prompts with renderers and calls
   a `SamplingClient`.
3. A simple eval runner to iterate samples and compute scores.
4. LLM-as-judge scoring that can parse `GRADE: C/I` outputs.

Everything else (tool calling, eval logging, multi-task loader) is optional
unless you want to run the full inspect-evals catalog.

