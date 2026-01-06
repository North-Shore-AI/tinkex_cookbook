# Critique Response and Plan Adjustments

Date: 2026-01-06

## Inputs Reviewed

- docs/20260106/plan_review/PLAN_VALIDATION_ANALYSIS.md
- docs/20260106/ecosystem_full_stack_plan/03_CRUCIBLE_KITCHEN_IMPLEMENTATION_PLAN.md
- docs/20260106/ecosystem_full_stack_plan/06_RECIPE_FOCUS_PLAN.md
- docs/20260106/ecosystem_full_stack_plan/08_REPO_IMPLEMENTATION_PLAN.md

## Summary of Critique

- Recipe focus mismatch: chat_sl, math_rl, distillation missing; code_rl exists but is out of scope.
- Adapter ownership drift: LLMClient, EmbeddingClient, VectorStore noops still in crucible_train.
- Snakebridge priority insufficient for math_rl.
- Missing explicit tiktoken_ex integration in repo plan.
- LLM SDK adapter migration plan not explicit.

## Decisions and Adjustments

1) Recipe alignment
- Keep the five focus recipes as-is.
- Derive math_rl from existing code_rl implementation.
- Implement chat_sl and distillation as new recipes and workflows.

2) Adapter ownership
- All adapters (including noops) live in crucible_kitchen.
- Tinkex_cookbook keeps zero adapters after migration.

3) Snakebridge priority
- Elevate to critical path for math_rl.
- Ship math_verify manifest and deterministic test scaffolding before math_rl parity work.

4) tiktoken_ex
- Add explicit tokenizer adapter integration tasks for kitchen and recipes.

5) LLM SDK adapters
- Migrate claude_agent_sdk, codex_sdk, gemini_ex adapters to crucible_kitchen.
- Provide uniform error, retry, and cost tracking semantics.

## Updated Execution Order

1) K0: Noop adapter migration + manifest updates
2) Snakebridge manifests and test harness
3) math_rl (fork code_rl + math_verify)
4) chat_sl (new supervised workflow extensions)
5) distillation (new workflow + sampling stages)
6) DPO parity validation
7) K3 eval + telemetry integration
8) K4 MLOps + control plane integration

## Acceptance Criteria for Adjustments

- No adapters remain in tinkex_cookbook.
- All adapter modules referenced in manifests are in crucible_kitchen.
- math_rl recipe runs with snakebridge math_verify.
- chat_sl and distillation run end-to-end through kitchen workflows.
- Parity tests pass for all five focus recipes.
