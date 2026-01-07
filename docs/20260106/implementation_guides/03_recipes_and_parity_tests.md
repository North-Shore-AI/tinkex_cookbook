# Focus Recipes + Parity/Property Tests

Date: 2026-01-06

## Scope

Implement missing recipes (chat_sl, math_rl, distillation) and add parity/
property tests for all five focus recipes:

1) sl_basic
2) chat_sl
3) preference (DPO)
4) math_rl
5) distillation

## Recipe Implementation Notes

### chat_sl (Supervised)

Target module:

- `tinkex_cookbook/lib/tinkex_cookbook/recipes/chat_sl.ex`

Key behavior:

- Uses `CrucibleKitchen.Workflows.Supervised`
- Dataset mapping for chat schemas (Tulu3, NoRobots, JSONL)
- Uses renderer name inferred from model
- Passes `train_on` and `max_length`

Implementation details:

- Add config schema with `ChzEx.Schema`
- Use `HfDatasetsEx` dataset name for Tulu3
- Normalize message schema to `%{role, content}`
- Ensure `add_special_tokens: false` on tokenizer

### math_rl (Reinforcement)

Target module:

- `tinkex_cookbook/lib/tinkex_cookbook/recipes/math_rl.ex`

Approach:

- Fork `code_rl.ex` to `math_rl.ex`
- Swap environment to GSM8K and use snakebridge `math_verify`
- Keep RL workflow (`CrucibleKitchen.Workflows.Reinforcement`)

Implementation details:

- Build `EnvGroupBuilder` for GSM8K problems
- Use `CrucibleKitchen.Adapters.Snakebridge.MathVerify` for reward
- Ensure reward is deterministic and normalized
- Add config for:
  - `env: :gsm8k`
  - `reward_tolerance`, `normalize_answer`

### distillation

Target module:

- `tinkex_cookbook/lib/tinkex_cookbook/recipes/distillation.ex`

Approach:

- Use `CrucibleKitchen.Workflows.Distillation`
- Require `sampling_client` and `training_client`
- Accept `teacher_model` and `sampling_params`

Implementation details:

- Define dataset options (prompts-only datasets)
- Optional `teacher_checkpoint_path`
- Pass `distill_alpha` / `distill_temperature`

## Parity Tests

Add parity tests for each focus recipe with deterministic fixtures.

Recommended locations:

- `tinkex_cookbook/test/parity/`
  - `sl_basic_parity_test.exs`
  - `chat_sl_parity_test.exs`
  - `dpo_parity_test.exs`
  - `math_rl_parity_test.exs`
  - `distillation_parity_test.exs`

Strategies:

- Use tiny fixtures (2-4 samples)
- Fix seed to `42` or `0`
- Compare token sequences and loss metrics to Python reference
- Use fixtures committed as JSON with expected metrics

Python references:

- `tinker-cookbook/tinker_cookbook/recipes/sl_basic.py`
- `tinker-cookbook/tinker_cookbook/recipes/chat_sl/train.py`
- `tinker-cookbook/tinker_cookbook/preference/train_dpo.py`
- `tinker-cookbook/tinker_cookbook/recipes/math_rl/train.py`
- `tinker-cookbook/tinker_cookbook/rl/train.py`
- `tinker-cookbook/tinker_cookbook/distillation/train_on_policy.py`

## Property Tests

Add property tests for deterministic behavior:

1) **Renderer invariants**
   - BOS/EOS handling
   - Stop sequences
   - Token count monotonicity

2) **Dataset determinism**
   - PCG64 shuffle repeatability
   - Fixed batch ordering

Suggested locations:

- `tinkex_cookbook/test/renderers/*_property_test.exs`
- `tinkex_cookbook/test/datasets/*_property_test.exs`

## No-Network Test Rules

- Use mocks for all external clients (Tinkex, HF, Snakebridge)
- No network in tests
- Use deterministic seeds
- Avoid sleeps

## Done When

- All three missing recipes compile and run via kitchen
- Parity tests pass for all five recipes
- Property tests pass and cover determinism and renderer invariants
