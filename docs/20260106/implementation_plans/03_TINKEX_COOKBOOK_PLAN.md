# Tinkex Cookbook Implementation Plan

Date: 2026-01-06

## Objective

Complete the thin facade transition and implement the missing focus recipes using crucible_kitchen workflows.

## Structural Changes

- Remove remaining adapter modules in tinkex_cookbook.
- Ensure runtime facade calls crucible_kitchen only.
- Keep recipe modules, config schemas, and Mix tasks.

## Recipe Implementation Tasks

### sl_basic

- Keep as baseline for parity.
- Ensure uses kitchen workflow with updated manifests.
- Maintain parity tests and token dumps.

### chat_sl

- Implement new recipe module.
- Add config schema for dataset name, split, max_length, renderer.
- Build messages from dataset using HF fields (Tulu3-like schema).
- Use kitchen supervised workflow with ChatDatumBuilder stage.

### preference (DPO)

- Validate existing recipe uses kitchen Preference workflow.
- Add parity tests for DPO loss and metrics.
- Ensure comparison dataset mapping is deterministic.

### math_rl

- Fork from code_rl and update:
  - Environment set to GSM8K (or configured dataset).
  - Reward computed via snakebridge math_verify.
  - Dataset builder adjusted for math problems.
- Add config schema for reward parameters and verification options.

### distillation

- Implement new recipe module.
- Add config schema for teacher model, sampling params, distillation loss.
- Use kitchen distillation workflow.

## Runtime and CLI

- Update runtime manifests to use kitchen adapters only.
- Mix tasks call TinkexCookbook.Runtime.run/2 with config.
- Add help output consistent across recipes.

## Acceptance Criteria

- tinkex_cookbook contains only recipes/config/runtime/CLI.
- All five focus recipes run through crucible_kitchen.
- Parity tests cover sl_basic, chat_sl, dpo, math_rl, distillation.

## Out of Scope

- tool_use, code_rl, multiplayer_rl, rubric, verifiers_rl, vlm_classifier.
