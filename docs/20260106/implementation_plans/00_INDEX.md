# Implementation Plan Index

Date: 2026-01-06
Status: Active

Purpose: Detailed, actionable implementation plans responding to the plan validation critique and covering all project aspects end-to-end.

## Document Map

- 00_INDEX.md (this file)
- 01_CRITIQUE_RESPONSE.md
- 02_CRUCIBLE_KITCHEN_PLAN.md
- 03_TINKEX_COOKBOOK_PLAN.md
- 04_ADAPTERS_AND_SDK_GAPS.md
- 05_DATA_PIPELINE_AND_DATASETS_PLAN.md
- 06_EVAL_TELEMETRY_TESTING_PLAN.md
- 07_MLOPS_CONTROL_PLANE_PLAN.md
- 08_RISK_TRACKING_AND_GATES.md

## Scope

- Full stack ML experiments platform with crucible_kitchen as thick core.
- tinkex_cookbook as thin recipe layer.
- Five focus recipes only: sl_basic, chat_sl, preference (DPO), math_rl, distillation.
- Adapter ownership per corrected model.
- End-to-end run lifecycle, telemetry, and parity testing.

Out of scope:
- Porting remaining tinker-cookbook recipes.
- Rewriting external SDKs.
- UI rebuilds outside control plane endpoints.
