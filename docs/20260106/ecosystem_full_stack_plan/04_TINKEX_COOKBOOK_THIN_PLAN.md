# Tinkex Cookbook Thin Plan

Date: 2026-01-06

## Goal

Make tinkex_cookbook a thin recipe layer that delegates all orchestration to crucible_kitchen and all training logic to crucible_train.

## Target Responsibilities

- Recipe definitions and defaults
- ChzEx config schemas and CLI entrypoints
- Minimal dataset helpers that are recipe-specific only
- No training loops, renderers, or core types
- No adapters (adapters live in crucible_kitchen)

## Target Structure

```
lib/tinkex_cookbook/
  runtime/
    runtime.ex       # Facade that calls crucible_kitchen
    manifests.ex     # Default adapter manifests
  recipes/
    behaviour.ex
    sl_basic.ex
    chat_sl.ex
    preference_dpo.ex
    math_rl.ex
    distillation.ex
  configs/
    sl_basic_config.ex
    chat_sl_config.ex
    preference_config.ex
    math_rl_config.ex
    distill_config.ex
  mix/tasks/
    sl_basic.ex
    chat_sl.ex
    preference.ex
    math_rl.ex
    distill.ex
```

## Migration Steps

1) Remove duplicated training logic (renderers, types, loops, metrics) from tinkex_cookbook.
2) Replace direct tinkex calls with crucible_kitchen workflow calls.
3) Align config schemas with CrucibleIR.Experiment fields.
4) Update mix tasks to call TinkexCookbook.Runtime.run/2.
5) Keep parity tooling for the five focus recipes only.

## Recipe Contract

Each recipe must provide:
- recipe_name/0
- build_config/1
- build_experiment/1 -> CrucibleIR.Experiment
- validate_config/1

## Acceptance Criteria

- tinkex_cookbook LOC stays minimal and stable.
- No direct dependency on crucible_train internals beyond types in config.
- All five focus recipes run through crucible_kitchen.
- Parity tests pass for the five recipes.
