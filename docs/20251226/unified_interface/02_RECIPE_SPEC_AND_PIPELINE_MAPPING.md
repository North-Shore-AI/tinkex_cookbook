# Recipe Spec and Pipeline Mapping

Date: 2025-12-26
Status: Draft
Owner: North-Shore-AI

## 1) Purpose

Define how recipes map to `CrucibleIR.Experiment` and which stages are used. This ensures a consistent spec-driven pipeline for all cookbook workflows.

## 2) Canonical Spec Shape

All recipes must emit a `CrucibleIR.Experiment` with:

- `id`: stable recipe id (atom)
- `pipeline`: list of `CrucibleIR.StageDef`
- `backend`: optional (for eval or inference stages)
- `training_config`: optional (for training stages)
- `dataset`: optional (if needed outside training config)

## 3) Stage Mapping by Recipe Type

### Supervised (sl_basic, sl_loop, chat_sl)

Pipeline stages:
1) `CrucibleTrain.Stages.SupervisedTrain`
2) Optional: `CrucibleModelRegistry.Stages.Register`
3) Optional: `CrucibleDeployment.Stages.Deploy`

Stage options for supervised training:

```elixir
%{
  training_config: %CrucibleIR.Training.Config{},
  ports: ports,
  log_path: "...",
  logger: {:module, state}
}
```

### RL (rl_basic, rl_loop, code_rl, math_rl)

Pipeline stages:
1) `CrucibleTrain.Stages.RLTrain`
2) Optional: register/deploy

Stage options must include:
- `training_config` (IR)
- `ports`
- RL specific options in `training_config.options`

### Preference (dpo)

Pipeline stages:
1) `CrucibleTrain.Stages.DPOTrain`
2) Optional: register/deploy

### Distillation (prompt_distillation, on_policy_distillation)

Pipeline stages:
1) `CrucibleTrain.Stages.Distillation`
2) Optional: register/deploy

## 4) Spec Construction Rules

- Use `CrucibleIR.Training.Config` as the canonical training config.
- Recipe-specific options go into `training_config.options`.
- Ports are resolved by the facade and injected into stage options.
- Any external integration (registry, deploy, feedback) is represented as an optional stage.

## 5) Example: sl_basic

```elixir
%CrucibleIR.Experiment{
  id: :sl_basic,
  pipeline: [
    %CrucibleIR.StageDef{
      name: :supervised_train,
      module: CrucibleTrain.Stages.SupervisedTrain,
      options: %{
        training_config: training_config,
        ports: ports,
        log_path: log_path
      }
    }
  ],
  training_config: training_config
}
```

## 6) Evaluation Integration

Eval can run as:
- A post-run step in the facade (not a stage), or
- A dedicated cookbook stage that wraps EvalEx.

For parity with `tinker_cookbook`, prefer facade post-run evaluation with:
- `TinkexCookbook.Eval.Runner`
- EvalEx tasks and scorers

## 7) Acceptance Criteria

- Every recipe emits an IR spec.
- Every recipe uses a `CrucibleTrain` stage for training.
- Optional MLOps stages are consistent across recipes.
