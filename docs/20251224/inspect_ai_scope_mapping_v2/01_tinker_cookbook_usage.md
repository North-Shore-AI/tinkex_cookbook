# tinker-cookbook inspect-ai Usage (Line-Referenced)

All inspect-ai usage in the Python cookbook lives under `tinker_cookbook/eval/`,
with one recipe wiring point for inline evals. Documentation references are
listed separately.

## Runtime Code (inspect-ai integration)

- `tinker-cookbook/tinker_cookbook/eval/inspect_utils.py:13-22`
  imports inspect-ai model and tool types used by the adapter.
- `tinker-cookbook/tinker_cookbook/eval/inspect_utils.py:30-42`
  `get_model_usage/2` builds `ModelUsage` from prompt + sampled tokens.
- `tinker-cookbook/tinker_cookbook/eval/inspect_utils.py:45-55`
  `convert_inspect_messages/1` only accepts `str` content and rejects
  non-text `Content` parts.
- `tinker-cookbook/tinker_cookbook/eval/inspect_utils.py:57-156`
  `InspectAPIFromTinkerSampling` implements inspect-ai `ModelAPI`:
  - wraps `tinker.SamplingClient.sample_async` (lines 123-128)
  - builds prompt via renderer + tokenizer (lines 95-121)
  - returns `ModelOutput` with `ChatCompletionChoice` + `ModelUsage` (lines 147-156)
  - accepts `tools` and `tool_choice` but does not use them (signature at 100-106)

- `tinker-cookbook/tinker_cookbook/eval/inspect_evaluators.py:7-106`
  `InspectEvaluator` wraps inspect-ai `eval_async` with:
  - `Tasks` (line 26)
  - `Model(api=..., config=GenerateConfig)` (lines 76-84)
  - concurrency/logging parameters (lines 87-106)

- `tinker-cookbook/tinker_cookbook/eval/custom_inspect_task.py:13-68`
  custom task definition using:
  - `@task` decorator (line 50)
  - `Task` + `MemoryDataset` + `Sample` (lines 13-33, 58-68)
  - `generate()` solver (line 61)
  - `model_graded_qa()` scorer (line 62)

- `tinker-cookbook/tinker_cookbook/eval/run_inspect_evals.py:11-47`
  CLI entrypoint that constructs `InspectEvaluator` and runs metrics.

## Recipe Wiring (inline evals)

- `tinker-cookbook/tinker_cookbook/recipes/chat_sl/train.py:88-108`
  inline evals use `InspectEvaluatorBuilder` with:
  - `tasks=["inspect_evals/gsm8k", "inspect_evals/ifeval"]` (line 97)
  - `renderer_name` and `model_name` (lines 98-99)

## Support Types (non inspect-ai, but used by evaluators)

- `tinker-cookbook/tinker_cookbook/eval/evaluators.py:10-30`
  defines `SamplingClientEvaluator` and `EvaluatorBuilder` interfaces used by
  `InspectEvaluatorBuilder`.

## Documentation References (inspect-ai usage)

- `tinker-cookbook/tinker_cookbook/eval/README.md:1`
  mentions inspect-ai integration and `run_inspect_evals.py`.
- `tinker-cookbook/docs/evals.mdx:32-119`
  describes offline evals via inspect-ai and includes the custom task example.
- `tinker-cookbook/docs/preferences/dpo-guide.mdx:93-102`
  shows running inspect evals for DPO models.
- `tinker-cookbook/llms-full.txt:535-568`
  mirrors the inspect-ai offline eval instructions.

## No Other Runtime Uses Found

- `rg -n "inspect_ai" tinker-cookbook/tinker_cookbook` only hits the eval files
  listed above.

