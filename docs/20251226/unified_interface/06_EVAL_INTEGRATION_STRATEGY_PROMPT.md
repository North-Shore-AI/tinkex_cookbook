# Prompt: Implement Eval Integration

Date: 2025-12-26

## Goal

Implement the eval integration described in:
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/unified_interface/06_EVAL_INTEGRATION_STRATEGY.md`

## Required Reading (Full Paths)

- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251226/unified_interface/06_EVAL_INTEGRATION_STRATEGY.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251223/INSPECT_AI_ELIXIR_ARCHITECTURE.md`
- `/home/home/p/g/North-Shore-AI/tinkex_cookbook/docs/20251224/inspect_ai_scope_mapping_v2/03_elixir_mapping_and_gaps.md`
- `/home/home/p/g/North-Shore-AI/crucible_harness/lib/research_harness/generate.ex`
- `/home/home/p/g/North-Shore-AI/eval_ex/lib/eval_ex/task.ex`
- `/home/home/p/g/North-Shore-AI/eval_ex/lib/eval_ex/scorer.ex`

## Context Summary

Evaluation is a thin adapter layer in `tinkex_cookbook` that wires EvalEx and CrucibleHarness to Tinkex sampling. It should remain minimal.

## Implementation Requirements

1) Ensure `TinkexCookbook.Eval.TinkexGenerate` implements `CrucibleHarness.Generate`.
2) Ensure `TinkexCookbook.Eval.Runner` wires tasks and scorers.
3) Add tests for eval wiring.

## TDD and Quality Gates

- Write tests first.
- `mix test` must pass.
- `mix compile --warnings-as-errors` must be clean.
- `mix format` must be clean.
- `mix credo --strict` must be clean.
- `mix dialyzer` must be clean.

## Version Bump (Required)

- Bump version `0.x.y` in `/home/home/p/g/North-Shore-AI/tinkex_cookbook/mix.exs`.
- Update `/home/home/p/g/North-Shore-AI/tinkex_cookbook/README.md` to reflect the new version.
- Add a 2025-12-26 entry to `/home/home/p/g/North-Shore-AI/tinkex_cookbook/CHANGELOG.md`.

