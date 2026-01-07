# Changelog (2026-01-06)

- Switched runtime manifests to kitchen-owned noops and removed cookbook adapter modules/tests.
- Trimmed SDK dependencies now owned by crucible_kitchen.
- Skipped SnakeBridge generation in tests to avoid Python provisioning.
- Updated AGENTS task tracker to reflect TrainingClient port completion.
- Updated AGENTS task tracker for Phase 3 RL stage progress.
- Added implementation guides for DPO refactor, distillation workflow, and recipe/parity tests.
- Updated AGENTS task tracker for current DPO/distillation/recipe milestones.
- Added dependency audit and integration plan doc.
- Updated AGENTS task tracker for dependency audit completion.
- Added chat_sl, math_rl, and distillation recipes wired to CrucibleKitchen workflows.
- Updated DPO recipe to require SamplingClient and added sampling adapter wiring.
- Added parity tests for all five focus recipes plus renderer/dataset property tests.
- Added StreamData test dependency for property-style coverage.
