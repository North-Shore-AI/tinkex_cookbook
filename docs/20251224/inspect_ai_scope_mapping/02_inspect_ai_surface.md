# inspect-ai Surface Used by tinker-cookbook (Line-Referenced)

This file documents the inspect-ai classes/functions that tinker-cookbook
actually uses, with source references for each definition.

## Eval Runner and Task Registry

- `inspect_ai/_eval/eval.py:282-326`
  `eval_async(...) -> list[EvalLog]` main async eval entrypoint.
- `inspect_ai/_eval/task/tasks.py:6-22`
  `Tasks` type alias, supports strings, Task objects, callables, and lists.
- `inspect_ai/_eval/task/task.py:59-183`
  `Task` class (dataset, solver, scorer, model, config, limits, etc).
- `inspect_ai/_eval/registry.py:87-139`
  `@task` decorator for registering tasks.

## Dataset Types

- `inspect_ai/dataset/_dataset.py:28-106`
  `Sample` class (input, target, choices, metadata, sandbox, etc).
- `inspect_ai/dataset/_dataset.py:240-313`
  `MemoryDataset` class.

## Solver + Scorer

- `inspect_ai/solver/_solver.py:267-293`
  `generate(...)` solver (default solver, tool-call loop in inspect-ai).
- `inspect_ai/scorer/_model.py:86-152`
  `model_graded_qa(...)` LLM-as-judge scorer.

## Model API + Types

- `inspect_ai/model/_registry.py:30-75`
  `modelapi(name)` decorator (registers ModelAPI implementations).
- `inspect_ai/model/_model.py:129-236`
  `ModelAPI` abstract class (requires `generate/4` with tools + tool_choice).
- `inspect_ai/model/_model.py:320-415`
  `Model` wrapper (holds ModelAPI + GenerateConfig, exposes generate).
- `inspect_ai/model/_generate_config.py:158-215`
  `GenerateConfig` (temperature, max_tokens, top_p, top_k, num_choices, etc).
- `inspect_ai/model/_chat_message.py:20-146`
  `ChatMessage` base types, including `ChatMessageSystem` and
  `ChatMessageAssistant`.
- `inspect_ai/_util/content.py:191-200`
  `Content` union used for multi-part messages (text/images/etc).
- `inspect_ai/model/_model_output.py:12-175`
  `ModelUsage`, `ChatCompletionChoice`, and `ModelOutput`.

## Tool Calling Types

- `inspect_ai/tool/_tool_choice.py:5-13`
  `ToolChoice` union type and `ToolFunction`.
- `inspect_ai/tool/_tool_info.py:19-54`
  `ToolInfo` (JSON Schema function descriptor).

## Minimal Field Usage in tinker-cookbook

The tinker-cookbook adapter uses only a small subset of the above:

- `GenerateConfig`: `temperature`, `max_tokens`, `top_p`, `top_k`,
  `num_choices`, and `system_message`.
- `ModelAPI.generate`: receives `tools` and `tool_choice`, but they are not
  used by `InspectAPIFromTinkerSampling`.
- `ChatMessage` content is assumed to be a `str` (not a Content list).
- `ModelUsage` is populated with counts derived from tokens from
  `tinker.SamplingClient`.

