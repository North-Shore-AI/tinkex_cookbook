# Remaining Work After Datasets

**Date:** 2025-12-21
**Updated:** 2025-12-23
**Prerequisites:** Dataset loading infrastructure complete (via hf_hub_ex + hf_datasets_ex)
**Python Reference:** ./tinker-cookbook/

## Executive Summary

**Status Update (2025-12-23):** All major library mappings are now complete:
- `pydantic` → `sinter` ~> 0.0.1 (added to mix.exs)
- `huggingface_hub` → `hf_hub_ex` (distinct from `hf_datasets_ex`)
- BEAM-obviated: asyncio, cloudpickle, tqdm, threading (no packages needed)
- See `docs/20251223/PYTHON_TO_ELIXIR_LIBRARY_MAPPING.md` for complete reference.

With datasets addressed, the remaining work falls into 4 major categories:

| Category | Effort | Priority | Blocker Level |
|----------|--------|----------|---------------|
| **chz Configuration Port** | 3-5 days | CRITICAL | Blocks ALL recipes |
| **Math Grading Stack** | 1 week | HIGH | Blocks math_rl only |
| **Recipe-Specific Deps** | 1-2 weeks | MEDIUM | Blocks 3 recipes |
| **Testing & Integration** | 1 week | HIGH | Quality gate |
| **Total** | **3-4 weeks** | - | - |

**Current mix.exs includes (updated 2025-12-23):**

Core training:
- `{:tinkex, "~> 0.3.2"}` - Tinker API (maps to `tinker`)
- `{:chz_ex, "~> 0.1.2"}` - configuration (maps to `chz`)
- `{:snakebridge, "~> 0.3.0"}` - Python interop (sympy/pylatexenc/math_verify)

Evaluation ecosystem (NEW):
- `{:crucible_harness, "~> 0.3.1"}` - Solver/Generate/TaskState (inspect-ai parity)
- `{:eval_ex, "~> 0.1.1"}` - Task/Sample/Scorer framework
- `{:crucible_datasets, "~> 0.5.1"}` - dataset management + MemoryDataset

Data + utilities:
- `{:hf_datasets_ex, "~> 0.1"}` - HuggingFace datasets loading
- `{:hf_hub, "~> 0.1"}` - HuggingFace Hub API
- `{:sinter, "~> 0.0.1"}` - validation (maps to `pydantic`)
- `{:nx, "~> 0.9"}` - tensors (maps to numpy/torch)
- `{:table_rex, "~> 4.0"}` - terminal tables (maps to `rich`)
- `{:ex_aws, "~> 2.5"}` + `{:ex_aws_s3, "~> 2.5"}` - cloud storage (maps to `blobfile`)

## 1. Non-Dataset Dependencies

### A. chz Configuration System (CRITICAL - 3-5 days)

**Status:** chz_ex v0.1.2 is available on Hex; still blocks until configs are ported.

**Files Affected:** 60 Python files use `@chz.chz` decorator

**What It Does:**
- Runtime-validated configuration dataclasses
- Field validation, munging (transformation), defaults
- Serialization to/from dict for checkpointing
- Type coercion and error messages

**Elixir Mapping (ChzEx):**
```
Python @chz.chz              →  use ChzEx.Schema + chz_schema
chz.field(default=x)         →  field :name, :type, default: x
chz.field(munger=fn)         →  field :name, :type, munger: fn
chz.asdict(config)           →  ChzEx.asdict(config)
chz.from_dict(dict, Class)   →  ChzEx.make(Class, dict)
chz.entrypoint(cls)          →  ChzEx.entrypoint(cls)
```

**Work Items:**
- [ ] Add `{:chz_ex, "~> 0.1.2"}` to `mix.exs`
- [ ] Add `{:snakebridge, "~> 0.3.0"}` to `mix.exs` and align math libs with manifests
- [ ] Define a config convention using `ChzEx.Schema` (field types + docs)
- [ ] Port supervised learning configs (4 classes)
- [ ] Port RL configs (6 classes)
- [ ] Port dataset builder configs (15+ classes)
- [ ] Port preference/DPO configs (3 classes)
- [ ] Replace serialization with `ChzEx.asdict/1` and `ChzEx.make/2`

**Estimated:** 3-5 days (mechanical but 60+ classes to port)

### B. Math Verification Stack (HIGH - 1 week)

**Status:** BLOCKS math_rl recipe ONLY

**Components:**
1. **sympy** - Symbolic math verification
2. **pylatexenc** - LaTeX to text parsing
3. **math-verify** - Answer equivalence checking

**Elixir Strategy:** Snakepit wrapper (Python process pool)

**Work Items:**
- [ ] Set up Snakepit process pool (1 day)
- [ ] Port math_grading.py functions (549 lines → wrapper, 2 days)
- [ ] Add timeout handling
- [ ] Test numerical equivalence against Python (1 day)

**Estimated:** 1 week

### C. scipy.signal.lfilter (TRIVIAL - 2 hours)

**Status:** Used ONLY for discounted rewards in RL

**Native Elixir:**
```elixir
def discounted_future_sum(rewards, gamma) when is_list(rewards) do
  rewards
  |> Enum.reverse()
  |> Enum.reduce({[], 0.0}, fn reward, {acc, future_sum} ->
    current_sum = reward + gamma * future_sum
    {[current_sum | acc], current_sum}
  end)
  |> elem(0)
end
```

**Estimated:** 2 hours

### D. Utility Libraries (TRIVIAL - 4-6 hours)

| Python | Elixir | Usage | Effort |
|--------|--------|-------|--------|
| `rich` (tables) | `TableRex` or `IO.ANSI` | Progress display | 1-2 hours |
| `termcolor` | `IO.ANSI` (built-in) | Colored output | 15 min |
| `anyio`/`asyncio` | `Task.async_stream/2` | Concurrency | 1 hour |
| `blobfile` (3 uses) | `File` + `ExAws.S3` | JSONL reads | 2 hours |

**Estimated:** 4-6 hours total

### E. Evaluation Stack Wiring (NEW - 1 day)

**Status:** Dependencies added (crucible_harness, eval_ex, crucible_datasets)

**What's Needed:**
- `TinkexCookbook.Eval.TinkexGenerate` - Adapter implementing `CrucibleHarness.Generate`
- `TinkexCookbook.Eval.Runner` - Wires EvalEx.Task + CrucibleHarness.Solver

**Work Items:**
- [ ] Create `lib/tinkex_cookbook/eval/tinkex_generate.ex` (~50 LOC)
- [ ] Create `lib/tinkex_cookbook/eval/runner.ex` (~50 LOC)
- [ ] Add tests for eval integration

**Estimated:** 1 day (thin wiring layer only)

### F. Optional Dependencies (SKIP UNLESS REQUESTED)

| Dependency | Recipe | Decision | If Needed |
|------------|--------|----------|-----------|
| textarena | multiplayer_rl | SKIP | Pythonx wrapper (3-5 days) |
| verifiers | verifiers_rl | SKIP | Edge case |
| chromadb | tool_use/search | Defer | `chroma` HTTP client (1-2 days) |
| google-genai | tool_use/search | Defer | HTTP client (1 day) |

## 2. Recipe-by-Recipe Gaps

### ✅ READY (6 recipes) - Just need chz port

| Recipe | Dependencies Met | Blocked By |
|--------|------------------|------------|
| `sl_basic` | tinkex | chz |
| `rl_basic` | tinkex | chz |
| `preference/dpo` | tinkex | chz |
| `code_rl` | tinkex, datasets | chz |
| `multiplayer_rl` (basic) | tinkex | chz |
| `verifiers_rl` (basic) | tinkex | chz |

### ⚠️ NEEDS WORK (4 recipes)

| Recipe | Blocked By | Additional Needs |
|--------|------------|------------------|
| sl_loop / rl_loop | chz | Nx handles torch ops |
| chat_sl | chz | Dataset wiring to crucible_datasets v0.4.1 |
| math_rl | chz + math grading | Snakepit math stack |
| tool_use/search | chz + chromadb + google-genai | `chroma` + HTTP client wrappers |

## 3. What's Blocking Each Recipe

| Recipe | Tinker API | chz | Datasets | Math | Other | Ready After |
|--------|-----------|-----|----------|------|-------|-------------|
| sl_basic | ✅ | ❌ | - | - | - | chz |
| sl_loop | ✅ | ❌ | ✅ | - | - | chz |
| rl_basic | ✅ | ❌ | - | - | - | chz |
| rl_loop | ✅ | ❌ | ✅ | - | scipy ❌ | chz+scipy |
| chat_sl | ✅ | ❌ | ✅ | - | - | chz |
| math_rl | ✅ | ❌ | ✅ | ❌ | - | chz+math |
| preference/dpo | ✅ | ❌ | - | - | - | chz |
| tool_use/search | ✅ | ❌ | - | - | ❌ | chz+tools |
| multiplayer_rl | ✅ | ❌ | - | - | ⚠️ | chz |
| code_rl | ✅ | ❌ | ✅ | - | - | chz |
| verifiers_rl | ✅ | ❌ | - | - | ⚠️ | chz |

## 4. Implementation Roadmap (Updated 2025-12-23)

### Week 1: Core Infrastructure + Eval Stack
- [ ] **Days 1-3:** Port remaining chz configs (many already done)
- [ ] **Day 4:** Add scipy.signal.lfilter native implementation + eval stack wiring
- [ ] **Day 5:** Concrete Llama3 renderer + renderer parity tests

**Deliverable:** 6 recipes ready, eval stack wired

### Week 2: Math Grading
- [ ] **Day 1:** Set up Snakepit process pool
- [ ] **Days 2-3:** Wrap sympy/pylatexenc/math-verify
- [ ] **Day 4:** Port math_grading.py logic
- [ ] **Day 5:** Test against Python outputs

**Deliverable:** math_rl recipe ready

### Week 3: Dataset Wiring + Training Integration
- [ ] **Day 1:** Wire NoRobots dataset builder with crucible_datasets
- [ ] **Days 2-3:** Complete Tinkex.TrainingClient integration in SlBasic
- [ ] **Day 4:** Wire `HF_TOKEN` and streaming options in dataset builders
- [ ] **Day 5:** Test chat_sl, sl_loop, rl_loop recipes

**Deliverable:** sl_loop, rl_loop, chat_sl recipes ready with actual training

### Week 4: Optional & Polish
- [ ] **Days 1-2:** Tool use wrappers if needed
- [ ] **Days 3-4:** End-to-end testing
- [ ] **Day 5:** Documentation and examples

**Deliverable:** All 11 recipes functional

## 5. Critical Path

```
tinkex SDK ✅ (DONE)
    ↓
Eval ecosystem ✅ (DONE - crucible_harness, eval_ex, crucible_datasets added)
    ↓
Concrete renderers + chz configs (Week 1)
    ├─→ 6 recipes READY
    ├─→ Eval stack wired
    ↓
Math grading (Week 2)
    ├─→ math_rl READY
    ↓
Dataset wiring + Training integration (Week 3)
    ├─→ sl_loop, rl_loop, chat_sl, code_rl READY
    ↓
Tool wrappers (Week 4, optional)
    ├─→ tool_use/search READY
    ↓
ALL 11 RECIPES COMPLETE + EVAL HARNESS
```

---

**Document Status:** Complete
**Last Updated:** 2025-12-23
