# Inspect AI Scope Mapping (2025-12-24)

Purpose: map how inspect-ai is used in tinker-cookbook and how that maps to
CrucibleHarness, EvalEx, and CrucibleBench.

## Sources Reviewed

- Python tinker-cookbook
  - `tinker-cookbook/tinker_cookbook/eval/inspect_utils.py`
  - `tinker-cookbook/tinker_cookbook/eval/inspect_evaluators.py`
  - `tinker-cookbook/tinker_cookbook/eval/custom_inspect_task.py`
  - `tinker-cookbook/tinker_cookbook/eval/run_inspect_evals.py`
  - `tinker-cookbook/tinker_cookbook/recipes/chat_sl/train.py` (inline evals)
- inspect-ai source in `inspect_ai/`
- Elixir libs in `../`
  - `crucible_harness` @ `f56f0f8` (Release v0.3.1)
  - `eval_ex` @ `272ab3a` (Release v0.1.1)
  - `crucible_bench` @ `4e11146` (feat bench stage + CrucibleIR config)

## Scope Summary

- inspect-ai usage in tinker-cookbook is limited to evaluation tooling.
- No inspect-ai usage in the core training loops or dataset builders.
- Inline inspect-ai evals are only wired into `recipes/chat_sl/train.py`.
- Tasks referenced in inline evals are `inspect_evals/gsm8k` and
  `inspect_evals/ifeval`, which live in the external inspect_evals repo.

## tinker-api Scan

- No `tinker-api` directory found under this repo or its parent.
- `rg` for `inspect_ai|inspect-ai|inspect ai` under `../tinker` returned no hits.
  If `tinker-api` exists elsewhere, re-run the scan from its root.

