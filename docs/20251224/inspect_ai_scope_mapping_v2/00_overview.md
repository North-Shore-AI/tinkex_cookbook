# Inspect AI Scope Mapping (tinker-cookbook only)

Purpose: re-scan the official Python cookbook at `./tinker-cookbook` for all
inspect-ai usage and map it to the Elixir stack.

## Repo Paths Clarified

- Python cookbook: `./tinker-cookbook` (source of inspect-ai integration).
- Python client used by cookbook: `tinker` package (imported in cookbook files).
- Elixir port of client: `../tinkex` (not the source of inspect-ai usage).
- Elixir eval stack:
  - `../crucible_harness` (LLM solver/generate/task state)
  - `../eval_ex` (Task/Sample/Scorer)
  - `../crucible_datasets` (MemoryDataset)
  - `../crucible_bench` (benchmarking; not used by cookbook eval runtime)

## Latest Commits (Local)

- `crucible_harness`: `f56f0f8` (Release v0.3.1)
- `eval_ex`: `272ab3a` (Release v0.1.1)
- `crucible_bench`: `4e11146` (Stage module + CrucibleIR config)

## Summary

- inspect-ai usage is confined to `tinker_cookbook/eval/*` and to inline eval
  wiring in `tinker_cookbook/recipes/chat_sl/train.py`.
- No inspect-ai usage exists in the core training or dataset pipelines.
- `inspect_evals/*` task names are referenced in recipe config and docs, but
  the inspect-evals repo is not present in this tree.

