# Recipe Focus Plan

Date: 2026-01-06

## Scope

Only the recipes below are in scope for porting and parity work during this plan.

### 1) sl_basic

- Purpose: baseline supervised learning.
- Dataset: NoRobots (HF).
- Workflow: Supervised.
- Critical deps: hf_datasets_ex, tinkex TrainingClient, crucible_train renderers.
- Status: already parity verified; used as baseline for kitchen.

### 2) chat_sl

- Purpose: chat SFT on conversational datasets (e.g. Tulu3).
- Dataset: Tulu3 SFT or similar HF dataset.
- Workflow: Supervised with chat renderer.
- Critical deps: hf_datasets_ex, tokenizer, chat renderers, dataset field mapping.
- Required kitchen work: dataset adapters for chat schema and message extraction.

### 3) preference (DPO)

- Purpose: preference learning and DPO.
- Dataset: pairwise preference data (HelpSteer, UltraFeedback, Tulu-Preference).
- Workflow: DPO with comparison datums.
- Critical deps: tinkex forward_backward_custom (or equivalent), comparison types in crucible_train.
- Required kitchen work: BuildComparisons stage and DPO metrics logging.

### 4) math_rl

- Purpose: RL with reward based on math verification.
- Dataset: GSM8K or math reasoning dataset.
- Workflow: RL with rollouts and advantage computation.
- Critical deps: snakebridge math_verify, RL metrics (KL, entropy), tokenizer alignment.
- Required kitchen work: Rollout stage, reward evaluator, advantage computation stage.

### 5) distillation

- Purpose: distill a teacher model into a student model.
- Dataset: teacher sampled outputs or existing HF dataset.
- Workflow: Distillation with teacher sampling.
- Critical deps: SamplingClient adapter, distillation loss pipeline.
- Required kitchen work: Teacher sampling stage and distill datum builder.

## Out of Scope Recipes

- tool_use
- code_rl
- multiplayer_rl
- rubric
- verifiers_rl
- vlm_classifier
- prompt_distillation

These can be revisited after the five focus recipes are stable.
