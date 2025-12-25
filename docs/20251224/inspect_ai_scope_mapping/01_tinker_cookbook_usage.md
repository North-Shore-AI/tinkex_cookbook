# tinker-cookbook inspect-ai Usage (Line-Referenced)

This file lists every runtime reference to inspect-ai in the Python
tinker-cookbook and where it is wired into recipes.

## Eval Integration Files

- `tinker-cookbook/tinker_cookbook/eval/inspect_utils.py:13-22`
  imports inspect-ai model and tool types used by the adapter.
- `tinker-cookbook/tinker_cookbook/eval/inspect_utils.py:57-156`
  defines `InspectAPIFromTinkerSampling`, the ModelAPI wrapper that:
  - accepts inspect-ai `ChatMessage` input and optional system message
  - builds a renderer prompt and calls `tinker.SamplingClient.sample_async`
  - returns `ModelOutput` with `ChatCompletionChoice` and `ModelUsage`
- `tinker-cookbook/tinker_cookbook/eval/inspect_utils.py:30-42`
  computes `ModelUsage` from tokenized prompt + sampled sequences.

- `tinker-cookbook/tinker_cookbook/eval/inspect_evaluators.py:7-106`
  uses `inspect_ai.Tasks`, `inspect_ai.eval_async`, and `Model` to run evals.
  It builds `InspectAIModel(api=InspectAPIFromTinkerSampling, config=GenerateConfig)`
  and passes evaluation parameters to `eval_async`.

- `tinker-cookbook/tinker_cookbook/eval/custom_inspect_task.py:13-68`
  defines a custom task with:
  - `@task` decorator
  - `Task` (dataset + solver + scorer)
  - `MemoryDataset` + `Sample`
  - `generate()` solver
  - `model_graded_qa()` scorer (LLM-as-judge)

- `tinker-cookbook/tinker_cookbook/eval/run_inspect_evals.py:6-47`
  CLI wrapper around `InspectEvaluator` for standalone inspect-ai runs.

## Recipe Entry Points

- `tinker-cookbook/tinker_cookbook/recipes/chat_sl/train.py:88-110`
  uses `InspectEvaluatorBuilder` for inline evals when
  `inline_evals == "inspect"`. Tasks are:
  - `inspect_evals/gsm8k`
  - `inspect_evals/ifeval`

## No Other Runtime Uses Found

- `rg -n "inspect_ai" tinker-cookbook/tinker_cookbook` only hits the files above.
- Other recipes (e.g., `recipes/vlm_classifier/eval.py`) do not use inspect-ai.

