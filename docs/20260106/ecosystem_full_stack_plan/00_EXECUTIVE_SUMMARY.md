# Full Stack ML Experiments Platform Plan

Date: 2026-01-06
Status: Draft for implementation

## Purpose

Create a unified, full stack ML experiments platform that makes crucible_kitchen the thick orchestration core, keeps tinkex_cookbook thin, and exposes the broader NSAI + Crucible ecosystem (including flowstone, command, synapse, claude_agent_sdk, codex_sdk, gemini_ex).

This plan aligns with the corrected ownership model: crucible_kitchen owns orchestration and adapters, crucible_train owns training ports and types, and external SDKs (tinkex, hf_datasets_ex, hf_hub_ex) remain independent.

## Scope

Included repos and systems:
- Orchestration: crucible_kitchen, crucible_framework, crucible_ir
- Training: crucible_train, tinkex, tinkex_cookbook
- Data: hf_datasets_ex, hf_hub_ex, crucible_datasets, datasets_ex
- Eval: eval_ex, crucible_harness, crucible_bench
- Observability: crucible_telemetry, crucible_trace
- MLOps: crucible_model_registry, crucible_deployment, crucible_feedback
- Reliability: crucible_ensemble, crucible_hedging, crucible_adversary, crucible_xai
- Control plane: nsai_gateway, nsai_registry, nsai_sites, pilot
- Agents and automation: command, synapse
- Data orchestration: flowstone
- LLM SDKs: claude_agent_sdk, codex_sdk, gemini_ex
- Python bridge: snakebridge + snakepit
- Config and tokenization: chz_ex, tiktoken_ex

Out of scope:
- Full port of all tinker-cookbook recipes
- Rewriting SDK repos (tinkex, hf_datasets_ex, hf_hub_ex)
- Major UI/UX work outside required control plane endpoints

## Focus Recipes (3-5)

Only these recipes are in scope for porting and parity during this plan:
1) sl_basic (baseline supervised learning)
2) chat_sl (chat SFT, e.g. Tulu3)
3) preference (DPO)
4) math_rl (RL with math verification)
5) distillation (model distillation)

Other recipes remain out of scope for this document set.

## Success Criteria

- crucible_kitchen provides complete workflows for supervised, RL, DPO, and distillation.
- tinkex_cookbook is reduced to recipes + config + CLI (thin facade).
- Adapters live in crucible_kitchen and implement crucible_train ports.
- Tinkex SDK gaps that block recipes are resolved.
- End-to-end runs are reproducible with deterministic seeds and telemetry.
- Parity checks pass for the five focus recipes.

## Document Index

- 00_EXECUTIVE_SUMMARY.md (this file)
- 01_STATUS_ASSESSMENT.md
- 02_TARGET_ARCHITECTURE.md
- 03_CRUCIBLE_KITCHEN_IMPLEMENTATION_PLAN.md
- 04_TINKEX_COOKBOOK_THIN_PLAN.md
- 05_DEPENDENCY_INTEGRATION_MATRIX.md
- 06_RECIPE_FOCUS_PLAN.md
- 07_ROADMAP_GATES_AND_TESTING.md
- 08_REPO_IMPLEMENTATION_PLAN.md

## Review Documents

- ../plan_review/PLAN_VALIDATION_ANALYSIS.md - Validation of plan docs against current repo state

## Implementation Plan Documents

- ../implementation_plans/00_INDEX.md - Detailed continuation plans for all aspects
