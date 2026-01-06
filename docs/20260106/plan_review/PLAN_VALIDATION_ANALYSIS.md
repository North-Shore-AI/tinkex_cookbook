# Plan Validation Analysis

Date: 2026-01-06

This document reviews the ecosystem full stack plan docs against current repo state and priorities.

---

## 1. Recipe Focus Plan Review (06_RECIPE_FOCUS_PLAN.md)

### Current Recipe Inventory

| Recipe | In Plan (Focus) | In Repo | Status |
|--------|-----------------|---------|--------|
| sl_basic | Yes | Yes (`sl_basic.ex`, `sl_basic_v2.ex`) | Parity verified per docs |
| chat_sl | Yes | No | **Missing implementation** |
| preference (DPO) | Yes | Yes (`dpo.ex`) | Implemented, uses `PreferenceWorkflow` |
| math_rl | Yes | No | **Missing implementation** |
| distillation | Yes | No | **Missing implementation** |
| code_rl | No (out of scope) | Yes (`code_rl.ex`) | Implemented, uses `ReinforcementWorkflow` |

### Findings

1. **Alignment gap**: `code_rl` exists and is functional but explicitly out of scope. `math_rl` is in scope but does not exist. These share the same workflow (`ReinforcementWorkflow`), suggesting math_rl can be derived from code_rl.

2. **chat_sl**: Missing entirely. Requires chat renderer and message extraction adapters. Dependency on hf_datasets_ex chat schema is not yet implemented.

3. **distillation**: Missing entirely. Requires `LoadTeacherSampler` and `SampleTeacher` stages per doc 03, plus SamplingClient adapter.

### Recommendations

| Item | Action |
|------|--------|
| math_rl | Fork from `code_rl.ex`, swap environment from `:deepcoder` to `:gsm8k`, integrate snakebridge `math_verify` |
| chat_sl | Create new recipe using supervised workflow, add chat datum builder stage |
| distillation | Create new workflow `Distillation` in crucible_kitchen, add teacher sampling stages |
| code_rl | Retain for future scope; do not delete; useful as RL workflow reference |

### Suggested Adjustments to Focus List

The five focus recipes are reasonable. However:

- **Consider promoting code_rl to focus** if team bandwidth allows, since it's already implemented and exercises the RL workflow that math_rl needs. Alternatively, keep code_rl out of scope but reuse its workflow code for math_rl.

- **Prioritization order** (based on implementation effort):
  1. sl_basic (done)
  2. preference/DPO (done, needs parity testing)
  3. math_rl (medium - fork code_rl + snakebridge)
  4. chat_sl (medium - new datum builder)
  5. distillation (high - new workflow + sampling stages)

---

## 2. Adapter Ownership Review (03_CRUCIBLE_KITCHEN_IMPLEMENTATION_PLAN.md)

### Current State in `manifests.ex`

```elixir
# Production adapters (crucible_kitchen ownership - correct)
training_client: KitchenAdapters.Tinkex.TrainingClient
dataset_store: KitchenAdapters.HfDatasets.DatasetStore
hub_client: KitchenAdapters.HfHub.HubClient
blob_store: KitchenAdapters.Noop.BlobStore

# Still using crucible_train noops (ownership ambiguity)
llm_client: TrainAdapters.Noop.LLMClient
embedding_client: TrainAdapters.Noop.EmbeddingClient
vector_store: TrainAdapters.Noop.VectorStore
```

### Ownership Model Compliance

| Port | Doc 03 Owner | Actual Owner | Status |
|------|--------------|--------------|--------|
| TrainingClient | crucible_kitchen adapter | KitchenAdapters.* | Compliant |
| DatasetStore | crucible_kitchen adapter | KitchenAdapters.* | Compliant |
| HubClient | crucible_kitchen adapter | KitchenAdapters.* | Compliant |
| BlobStore | crucible_kitchen adapter | KitchenAdapters.* | Compliant |
| LLMClient | crucible_kitchen adapter | **TrainAdapters.*** | **Non-compliant** |
| EmbeddingClient | crucible_kitchen adapter | **TrainAdapters.*** | **Non-compliant** |
| VectorStore | crucible_kitchen adapter | **TrainAdapters.*** | **Non-compliant** |

### Deleted Adapters (per git status)

The following noop adapters were deleted from `tinkex_cookbook`:
- `adapters/blob_store/noop.ex`
- `adapters/dataset_store/hf_datasets.ex`
- `adapters/dataset_store/noop.ex`
- `adapters/embedding_client/noop.ex`
- `adapters/hub_client/hf_hub.ex`
- `adapters/hub_client/noop.ex`
- `adapters/llm_client/noop.ex`
- `adapters/training_client/noop.ex`
- `adapters/training_client/tinkex.ex`
- `adapters/vector_store/noop.ex`

This deletion is correct per the thin-cookbook migration. Adapters now live in crucible_kitchen.

### Remaining Adapters in tinkex_cookbook

These adapters still exist locally (should they be migrated?):
- `adapters/blob_store/local.ex`
- `adapters/llm_client/claude_agent.ex`
- `adapters/llm_client/codex.ex`
- `adapters/vector_store/chroma.ex`

### Findings

1. **Noop ownership drift**: LLMClient, EmbeddingClient, VectorStore noops still reference `CrucibleTrain.Adapters.Noop.*`. Per doc 03, all adapter implementations should live in crucible_kitchen.

2. **Local adapters**: Some specialized adapters (claude_agent, codex, chroma, local blob) remain in tinkex_cookbook. Decision needed: migrate to crucible_kitchen or keep as cookbook-specific extensions?

### Recommendations

| Item | Action |
|------|--------|
| Noop adapters | Create `KitchenAdapters.Noop.{LLMClient, EmbeddingClient, VectorStore}` in crucible_kitchen |
| Local adapters | Keep in tinkex_cookbook as optional extensions OR migrate if reusable across cookbooks |
| manifests.ex | Update to use `KitchenAdapters.Noop.*` once created |

---

## 3. Workflow Phases Review (03_CRUCIBLE_KITCHEN_IMPLEMENTATION_PLAN.md)

### Phase Status

| Phase | Description | Status | Gaps |
|-------|-------------|--------|------|
| K0 | Ownership alignment | Partial | Noop adapters still in crucible_train |
| K1 | Workflow completion | Partial | Supervised/Preference done; RL partial; Distillation missing |
| K2 | Adapter/SDK integration | Partial | Tinkex, HF done; LLM adapters, snakebridge pending |
| K3 | Evaluation/Telemetry | Not started | EvalEx integration, harness hooks pending |
| K4 | MLOps/Control plane | Not started | Registry, deployment, feedback integration pending |

### Critical Path

1. **K1 completion is blocking** - math_rl and distillation recipes cannot run without their workflows
2. **K2 snakebridge** - math_rl depends on `math_verify` Python bridge
3. **K3** - Can proceed in parallel once K1 is stable

---

## 4. Repo Implementation Plan Review (08_REPO_IMPLEMENTATION_PLAN.md)

### Priority Alignment Check

| Repo | Doc 08 Priority | Current Reality | Assessment |
|------|-----------------|-----------------|------------|
| crucible_kitchen | Core | Active development | Aligned |
| crucible_train | Core | Production-ready per status doc | Aligned |
| tinkex | Core | Mature but gaps | Aligned - needs endpoint coverage |
| tinkex_cookbook | Core | Phase 1 complete, thinning in progress | Aligned |
| hf_datasets_ex | Data | Mature | Aligned |
| crucible_datasets | Data | Mature | Aligned |
| eval_ex | Eval | Mature | Aligned |
| snakebridge | Bridge | Required for math_rl | **Needs prioritization** |
| flowstone | Data | Listed but not in focus recipes | Lower priority OK |
| command | Agent | Listed for HITL runs | Lower priority for training focus |
| synapse | Agent | Listed for multi-agent | Lower priority for training focus |

### Missing from Doc 08

- No explicit mention of `tiktoken_ex` integration timeline
- No explicit mention of `claude_agent_sdk` / `codex_sdk` adapter migration

### Recommendations

| Item | Action |
|------|--------|
| snakebridge | Elevate priority - blocking for math_rl |
| flowstone, command, synapse | Deprioritize until focus recipes stable |
| tiktoken_ex | Add section for tokenizer adapter integration |
| LLM SDKs | Add migration plan for claude_agent_sdk, codex_sdk adapters |

---

## 5. Cross-Document Consistency

### Verified Consistent

- Ownership model (kitchen owns adapters, train owns ports)
- Focus recipe list (same 5 across docs)
- Workflow definitions (Supervised, Preference, RL, Distillation)

### Inconsistencies Found

| Issue | Location | Resolution |
|-------|----------|------------|
| code_rl exists but out of scope | 06 vs repo | Clarify: keep for reference, extract RL workflow for math_rl |
| Noop adapter ownership | 03 vs manifests.ex | Migrate noops to crucible_kitchen |
| snakebridge priority | 08 | Elevate - math_rl blocker |

---

## 6. Summary of Required Actions

### High Priority (Blocking)

1. **Create math_rl recipe** - Fork code_rl, integrate snakebridge math_verify
2. **Create chat_sl recipe** - Add chat datum builder stage
3. **Create distillation recipe + workflow** - New workflow in crucible_kitchen
4. **Migrate noop adapters** - Move LLMClient, EmbeddingClient, VectorStore noops to crucible_kitchen

### Medium Priority (Supporting)

5. **DPO parity testing** - Recipe exists but needs verification
6. **snakebridge manifests** - Ensure math_verify, sympy, pylatexenc are ready
7. **Decide on local adapters** - Migrate or keep claude_agent, codex, chroma, local blob

### Lower Priority (Post-Focus)

8. **K3 Evaluation hookup** - EvalEx, CrucibleHarness integration
9. **K4 MLOps** - Registry, deployment, feedback pipelines
10. **Agent integration** - command, synapse run lifecycle

---

## 7. Conclusion

The plan documents are well-structured and internally consistent. The primary gap is between documented intent and current implementation state:

- **2 of 5 focus recipes implemented** (sl_basic, dpo)
- **3 of 5 focus recipes missing** (chat_sl, math_rl, distillation)
- **Adapter ownership 70% compliant** (noop migration pending)
- **Workflow coverage 50%** (Supervised, Preference done; RL partial; Distillation missing)

The plan is valid and actionable. Recommended execution order:

1. Complete K0 (noop adapter migration)
2. Implement math_rl (reuse code_rl + snakebridge)
3. Implement chat_sl (extend supervised workflow)
4. Implement distillation (new workflow)
5. Parity test all 5 recipes
6. Proceed to K3/K4
