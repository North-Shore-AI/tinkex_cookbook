# Data Pipeline and Datasets Plan

Date: 2026-01-06

## Goals

- Ensure deterministic dataset loading and batching across all focus recipes.
- Standardize dataset field mapping and message extraction.
- Maintain parity with Python tinker-cookbook data processing.

## Dataset Requirements by Recipe

### sl_basic

- Dataset: NoRobots
- Fields: messages or prompt/response depending on dataset split
- Requirements: deterministic shuffle, consistent max_length

### chat_sl

- Dataset: Tulu3 or compatible chat SFT
- Fields: messages array with roles and content
- Requirements: field mapping adapter for dataset schema

### preference (DPO)

- Dataset: HelpSteer / UltraFeedback / Tulu-Preference
- Fields: chosen/rejected pairs and prompt context
- Requirements: comparison builder with deterministic ordering

### math_rl

- Dataset: GSM8K (or other math sets)
- Fields: question, answer
- Requirements: environment builder produces problem prompts

### distillation

- Dataset: teacher-generated or static SFT dataset
- Fields: prompt/teacher_response or message list
- Requirements: teacher sampling path with prompt templates

## Implementation Tasks

- Define FieldMapping structs for each dataset family.
- Add dataset-specific builders in crucible_kitchen or crucible_train (where types live).
- Implement ChatDatumBuilder stage for chat_sl.
- Implement ComparisonDatumBuilder for DPO.
- Implement RL environment data adapter for math_rl.
- Implement DistillDatumBuilder for distillation.

## Determinism and Parity

- Enforce PCG64 shuffling for all dataset builders.
- Require explicit seeds in config.
- Add snapshot tests for dataset ordering and token counts.

## Data Lineage

- Record dataset refs and versions in run context.
- Emit dataset metadata to telemetry.
- Store dataset hash in model registry metadata.
