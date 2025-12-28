# TinkexCookbook Thin Facade Plan

**Date:** 2025-12-27
**Status:** Implementation Ready
**Target:** Transform tinkex_cookbook from 13,633 LOC standalone to <4,000 LOC thin facade

---

## Executive Summary

This document set provides a comprehensive plan for transforming `tinkex_cookbook` into a thin facade that wraps the North-Shore-AI Crucible ecosystem. The transformation leverages the recently decomposed libraries (Dec 2025) and integrates Snakepit 0.8.1 for Python bridging.

### Key Transformation Goals

1. **Eliminate Duplication**: Remove ~10,000 LOC duplicated in crucible_train
2. **Unified Facade**: Single entrypoint (`TinkexCookbook.Runtime`) for all recipe execution
3. **Hexagonal Architecture**: Ports + Adapters pattern for swappable backends
4. **Snakepit Integration**: Python libs (sympy, pylatexenc, math_verify) via gRPC bridge in recipes

---

## Document Set Index

| Document | Purpose |
|----------|---------|
| [00_OVERVIEW.md](./00_OVERVIEW.md) | This file - executive summary and index |
| [01_LIBRARY_INVENTORY.md](./01_LIBRARY_INVENTORY.md) | Complete inventory of required North-Shore-AI libraries |
| [02_FACADE_ARCHITECTURE.md](./02_FACADE_ARCHITECTURE.md) | Core facade design with diagrams |
| [03_SNAKEPIT_INTEGRATION.md](./03_SNAKEPIT_INTEGRATION.md) | Python bridge wiring for recipes |
| [04_IMPLEMENTATION_ROADMAP.md](./04_IMPLEMENTATION_ROADMAP.md) | Step-by-step implementation plan |

---

## Current State (Before)

```
tinkex_cookbook v0.3.1 (13,633 LOC)
├── renderers/          # DUPLICATED in crucible_train
├── types/              # DUPLICATED in crucible_train
├── supervised/         # DUPLICATED in crucible_train
├── rl/                 # DUPLICATED in crucible_train
├── preference/         # DUPLICATED in crucible_train
├── distillation/       # DUPLICATED in crucible_train
├── completers/         # DUPLICATED in crucible_train
├── utils/              # PARTIALLY duplicated
├── ports/              # Missing TrainingClient port!
├── adapters/           # Tinker-specific (KEEP)
├── recipes/            # Recipe orchestration (KEEP)
├── eval/               # Evaluation wiring (KEEP)
└── datasets/           # Recipe-specific loaders (KEEP)
```

**Critical Problem**: tinkex_cookbook does NOT depend on crucible_train despite crucible_train being built specifically for this purpose.

---

## Target State (After)

```
tinkex_cookbook v0.4.0 (<4,000 LOC)
├── runtime/            # Unified facade + composition root
│   ├── runtime.ex      # TinkexCookbook.Runtime (entrypoint)
│   └── manifests.ex    # Environment-specific port wiring
├── recipes/            # Recipe behaviour + implementations
│   ├── behaviour.ex    # Recipe behaviour definition
│   ├── sl_basic.ex     # Supervised learning recipe
│   ├── rl_grpo.ex      # RL recipe
│   ├── dpo.ex          # DPO recipe
│   └── distill.ex      # Distillation recipe
├── adapters/           # Tinker-specific adapter implementations
│   ├── training_client/
│   │   └── tinkex.ex   # Implements CrucibleTrain.Ports.TrainingClient
│   ├── dataset_store/
│   │   └── hf_datasets.ex
│   ├── hub_client/
│   │   └── hf_hub.ex
│   ├── blob_store/
│   │   ├── local.ex
│   │   └── s3.ex
│   ├── llm_client/
│   │   ├── claude_agent.ex
│   │   └── codex.ex
│   └── vector_store/
│       └── chroma.ex
├── cli/                # Mix tasks for recipe execution
│   └── sl_basic.ex
├── eval/               # EvalEx + inspect-ai integration
│   ├── runner.ex
│   └── tinkex_generate.ex
└── config/             # ChzEx schemas
    └── sl_basic_config.ex
```

---

## Dependency Layer Model

```
                    APPLICATION LAYER
┌─────────────────────────────────────────────────────┐
│                  tinkex_cookbook                     │
│  (Runtime + Recipes + Adapters + CLI)               │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
              TRAINING INFRASTRUCTURE
┌─────────────────────────────────────────────────────┐
│                  crucible_train                      │
│  (Types, Renderers, Training Loops, Logging, Ports) │
└─────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│crucible_model│ │  crucible_   │ │  crucible_   │
│  _registry   │ │  deployment  │ │  feedback    │
└──────────────┘ └──────────────┘ └──────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         ▼
               CORE FRAMEWORK LAYER
┌─────────────────────────────────────────────────────┐
│  crucible_framework    crucible_ir    crucible_bench│
│  (Pipeline Runner)     (Specs/IR)    (Statistics)   │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
                PYTHON BRIDGE LAYER
┌─────────────────────────────────────────────────────┐
│                     snakepit                         │
│  (gRPC Bridge, Session Management, Worker Pools)    │
└─────────────────────────────────────────────────────┘
```

---

## Non-Negotiable Invariants

1. **Recipes are orchestration only** - no direct client calls or training logic
2. **All external integrations go through ports/adapters** - swappable at runtime
3. **The facade is the only entrypoint** - `TinkexCookbook.Runtime.run/2`
4. **Training logic lives in crucible_train** - not duplicated in cookbook
5. **Snakepit calls happen in recipes** - not in the facade core
6. **IR types are the contract** - `CrucibleIR.Experiment` flows through the system

---

## Quick Reference: What Goes Where

| Component | Location | Notes |
|-----------|----------|-------|
| Renderers | crucible_train | Llama3, Qwen3, DeepSeek, etc. |
| Types (Datum, ModelInput) | crucible_train | Shared data structures |
| Training loops | crucible_train | SupervisedTrain, RLTrain, DPO stages |
| Port behaviours | crucible_train | TrainingClient, DatasetStore, etc. |
| Logging backends | crucible_train | W&B, Neptune, JSON loggers |
| Tinkex adapter | tinkex_cookbook | Implements TrainingClient for Tinkex |
| Recipe definitions | tinkex_cookbook | sl_basic, rl_grpo, dpo, distill |
| ChzEx configs | tinkex_cookbook | CLI config schemas |
| Python math tools | snakepit (via recipes) | sympy, pylatexenc, math_verify |
| Evaluation | eval_ex + crucible_harness | inspect-ai parity |

---

## Verification Checklist

After implementation, verify:

- [ ] `tinkex_cookbook` LOC < 4,000
- [ ] `tinkex_cookbook` depends on `crucible_train >= 0.2.0`
- [ ] No renderer/type/training code remains in tinkex_cookbook
- [ ] `mix tinkex.sl_basic` executes via facade
- [ ] Snakepit adapters work in math verification recipes
- [ ] All tests pass with mocked ports

---

## Related Documents

- `tinkex_cookbook/docs/20251226/thin_cookbook_design/THIN_COOKBOOK_FOUNDATION_DESIGN.md`
- `tinkerer/brainstorm/20251226/ECOSYSTEM_CONSOLIDATION_ROADMAP.md`
- `tinkerer/brainstorm/20251226/STAGE_DESCRIBE_CONTRACT.md`
- `snakepit/ARCHITECTURE.md`
