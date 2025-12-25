# inspect-ai Surface Used by tinker-cookbook (Line-Referenced)

This list focuses on the inspect-ai classes and functions invoked by
tinker-cookbook. It excludes unrelated inspect-ai subsystems.

## Eval Runner and Tasks

- `inspect_ai/_eval/eval.py:282-326`
  `eval_async(tasks, model, ...)` main async eval entrypoint.
- `inspect_ai/_eval/task/tasks.py:6-22`
  `Tasks` type alias (supports strings, Task objects, callables, and lists).
- `inspect_ai/_eval/task/task.py:59-183`
  `Task` class (dataset, solver, scorer, model, config, limits).
- `inspect_ai/_eval/registry.py:87-159`
  `@task` decorator (registers tasks + metadata).

## Dataset Types

- `inspect_ai/dataset/_dataset.py:28-105`
  `Sample` class (input, target, choices, metadata, sandbox, files).
- `inspect_ai/dataset/_dataset.py:240-320`
  `MemoryDataset` class.

## Solver + Scorer

- `inspect_ai/solver/_solver.py:267-293`
  `generate(...)` solver (tool-call loop options).
- `inspect_ai/scorer/_model.py:86-152`
  `model_graded_qa(...)` LLM-as-judge scorer.

## Model API + Types

- `inspect_ai/model/_registry.py:30-76`
  `modelapi(name)` decorator for registering `ModelAPI` classes.
- `inspect_ai/model/_model.py:129-223`
  `ModelAPI` abstract class (`generate/4` with tools + tool_choice).
- `inspect_ai/model/_model.py:320-415`
  `Model` wrapper (binds ModelAPI + GenerateConfig).
- `inspect_ai/model/_generate_config.py:158-216`
  `GenerateConfig` (temperature, max_tokens, top_p, top_k, num_choices, etc).
- `inspect_ai/model/_chat_message.py:20-153`
  `ChatMessage` base types (system/user/assistant roles).
- `inspect_ai/_util/content.py:191-200`
  `Content` union (text, reasoning, image, audio, tool use, etc).
- `inspect_ai/model/_model_output.py:12-175`
  `ModelUsage`, `ChatCompletionChoice`, `ModelOutput`.

## Tool Calling Types

- `inspect_ai/tool/_tool_choice.py:5-13`
  `ToolChoice` union and `ToolFunction`.
- `inspect_ai/tool/_tool_info.py:19-54`
  `ToolInfo` (JSON schema tool descriptor).

## Minimal Field Usage in tinker-cookbook

tinker-cookbook only uses a subset of the above:

- `GenerateConfig`: `temperature`, `max_tokens`, `top_p`, `top_k`,
  `num_choices`, `system_message`.
- `ModelAPI.generate` receives `tools` + `tool_choice` but the cookbook adapter
  ignores them.
- `ChatMessage.content` is assumed to be a plain `str` (multi-part `Content`
  is rejected).
- `ModelUsage` is constructed from tokenized prompt + sampled tokens.

