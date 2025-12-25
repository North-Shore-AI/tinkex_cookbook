# Elixir Mapping and Gap Analysis

This file maps the Python inspect-ai usage in tinker-cookbook to the Elixir
libraries (CrucibleHarness, EvalEx, CrucibleBench) and calls out gaps.

## Direct Mapping Table

Python usage (file:line) | inspect-ai API | Elixir equivalent | Coverage notes
---|---|---|---
`tinker-cookbook/tinker_cookbook/eval/inspect_utils.py:57-156` | `ModelAPI.generate(...)` + `ModelOutput` | `TinkexCookbook.Eval.TinkexGenerate.generate/2` (`lib/tinkex_cookbook/eval/tinkex_generate.ex:61-179`) + `CrucibleHarness.Generate` behaviour (`../crucible_harness/lib/research_harness/generate.ex:1-172`) | Partial: response fields align, but token usage is stubbed (prompt_tokens=0) and renderer parsing differs.
`tinker-cookbook/tinker_cookbook/eval/inspect_evaluators.py:68-106` | `eval_async(tasks, model, ...)` | `TinkexCookbook.Eval.Runner.run/2` and `run_task/2` (`lib/tinkex_cookbook/eval/runner.ex:76-266`) | Partial: no async concurrency, logging, task loading, or retry policy.
`tinker-cookbook/tinker_cookbook/eval/custom_inspect_task.py:50-68` | `@task` + `Task` | `EvalEx.Task` struct or behaviour (`../eval_ex/lib/eval_ex/task.ex:1-123`) | Partial: EvalEx has behaviour + struct, no decorator macro.
`tinker-cookbook/tinker_cookbook/eval/custom_inspect_task.py:21-33` | `MemoryDataset` + `Sample` | `EvalEx.Sample` (`../eval_ex/lib/eval_ex/sample.ex:1-93`) | Partial: no MemoryDataset in EvalEx; datasets are lists of samples. (CrucibleDatasets is separate.)
`tinker-cookbook/tinker_cookbook/eval/custom_inspect_task.py:61-62` | `generate()` solver | `CrucibleHarness.Solver.Generate.new/1` (`../crucible_harness/lib/research_harness/solver/generate.ex:1-182`) | Partial: no tool-call loop like inspect-ai.
`tinker-cookbook/tinker_cookbook/eval/custom_inspect_task.py:62-68` | `model_graded_qa(...)` | `EvalEx.Scorer.LLMJudge.score/2` (`../eval_ex/lib/eval_ex/scorer/llm_judge.ex:1-84`) | Partial: LLMJudge uses "CORRECT/INCORRECT" parsing, not inspect-ai's default grade pattern.
`tinker-cookbook/tinker_cookbook/eval/inspect_evaluators.py:76-84` | `Model(api=..., config=GenerateConfig)` | `TinkexCookbook.Eval.TinkexGenerate` config map + `CrucibleHarness.Generate` | Partial: no GenerateConfig struct, no model roles or per-model configs.
`tinker-cookbook/tinker_cookbook/eval/inspect_utils.py:13-22` | `ToolInfo`, `ToolChoice`, `ChatMessage` types | No direct equivalents in EvalEx/CrucibleHarness | Missing: tool calling types not implemented in Elixir libs.
`tinker-cookbook/tinker_cookbook/recipes/chat_sl/train.py:93-108` | inline eval tasks `inspect_evals/*` | No native task loader in EvalEx | Missing: task registry + ports for inspect_evals tasks (gsm8k, ifeval).

## CrucibleBench Fit

- `crucible_bench` provides statistical benchmarking (`../crucible_bench/lib/crucible_bench/stage.ex:1-200`)
  but does not correspond to inspect-ai runtime eval features used in tinker-cookbook.
- It could be used post-eval to compare metric distributions, but there is no
  call site in tinker-cookbook today.

## Gaps and Deltas (Required for Full Parity)

1. Task loading
   - inspect-ai supports `Tasks` as strings and loads from task registry/files.
   - EvalEx has a registry but no loader for inspect_evals-style task strings.

2. Async eval runner
   - inspect-ai `eval_async` handles concurrency, retries, log_dir, and limits.
   - `TinkexCookbook.Eval.Runner` is synchronous and has no logging pipeline.

3. Tool calling
   - `ModelAPI.generate` signature includes `tools` and `tool_choice`.
   - No Elixir equivalents for ToolInfo/ToolChoice or tool-call flows.

4. LLM-as-judge parity
   - inspect-ai `model_graded_qa` supports templates, grade regex, partial credit,
     and model roles (grader).
   - `EvalEx.Scorer.LLMJudge` is a minimal prompt + "CORRECT/INCORRECT" parser.

5. Usage accounting
   - inspect-ai returns `ModelUsage` with token counts from prompt + completion.
   - `TinkexGenerate.format_response/2` currently uses string length and sets
     prompt_tokens to 0.

6. Multi-part message content
   - inspect-ai `ChatMessage` supports `Content` lists (text/images/tools).
   - The adapter in `inspect_utils.py` only accepts string content. Elixir
     messaging types do not yet expose the Content union.

## What is Already Sufficient for the Current Recipes

- A minimal evaluator that:
  1. Builds a prompt from chat messages using a renderer,
  2. Calls a sampling client,
  3. Parses a single assistant response,
  4. Scores via exact match or LLM-judge.

This is aligned with:
- `TinkexCookbook.Eval.TinkexGenerate` (generation)
- `EvalEx.Task` + `EvalEx.Sample` (task + dataset)
- `EvalEx.Scorer.LLMJudge` or `EvalEx.Scorer.ExactMatch` (scoring)

