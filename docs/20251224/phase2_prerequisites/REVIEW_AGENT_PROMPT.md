# Phase 2 Part 1 Infrastructure Review Agent Prompt

**Purpose:** Critical review and correction of the Phase 2 Part 1 infrastructure document.

---

## Task

You are a senior Elixir engineer tasked with critically reviewing the Phase 2 Part 1 infrastructure document. Your job is to:

1. **Verify accuracy** by cross-referencing against the actual Python and Elixir codebases
2. **Identify gaps** - components mentioned but missing, or missing entirely
3. **Correct errors** - wrong LOC counts, incorrect type signatures, misunderstood patterns
4. **Update the document** with your corrections directly

## Decisions (2025-12-24)

These decisions are fixed and should be reflected in the infra doc:

1. **VL renderers deferred:** Qwen3VL/Qwen3VLInstruct are Phase 3, not Phase 2 Part 1.
2. **Tool calling framework:** Implement shared tool-call encode/decode at the framework level.
3. **Sync + async RL together:** Do not phase async later; build both paths together.

---

## Codebases to Review

### Python tinker-cookbook (reference implementation)
**Path:** `./tinker-cookbook/tinker_cookbook/`

Critical files to examine:
```
renderers.py              # All renderer implementations
completers.py             # TokenCompleter, MessageCompleter
checkpoint_utils.py       # Checkpointing logic
rl/types.py               # RL type definitions
rl/rollouts.py            # Rollout functions
rl/data_processing.py     # Advantage computation, data assembly
rl/train.py               # RL training loops
rl/metrics.py             # KL metrics, trajectory metrics
preference/types.py       # Preference types
preference/dpo_datasets.py # DPO dataset builders
preference/train_dpo.py   # DPO training logic
distillation/datasets.py  # Distillation data
distillation/train_on_policy.py # On-policy distillation
utils/lr_scheduling.py    # LR scheduling
utils/format_colorized.py # Display utilities
```

### Elixir tinkex_cookbook (current implementation)
**Path:** `./lib/tinkex_cookbook/`

Review what already exists:
```
renderers/                # Llama3 exists, check patterns
supervised/               # Training loop patterns
types/                    # Datum, ModelInput, TensorData
utils/                    # What utilities exist?
eval/                     # TinkexGenerate adapter
```

---

## Review Checklist

### 1. Renderer Analysis

- [ ] Count actual LOC for each Python renderer (not estimates)
- [ ] Verify renderer method signatures match document
- [ ] Confirm VL variants are explicitly deferred to Phase 3 and documented as such
- [ ] Confirm tool calling is documented as a shared framework module (not per-renderer copy/paste)
- [ ] Identify any special token handling not documented
- [ ] Verify Qwen3 thinking mode logic is accurately described
- [ ] Check DeepSeek separator characters (unicode)

**Questions to answer:**
- Are there renderers used by Phase 2 recipes that aren't listed?
- What's the actual complexity of each renderer?
- Are there shared utilities between renderers that should be extracted?

### 2. Completers Analysis

- [ ] Verify `completers.py` structure matches document
- [ ] Check if there are additional completer types not documented
- [ ] Verify stop condition types are accurate
- [ ] Check how logprobs are handled

### 3. RL Infrastructure Analysis

- [ ] Read `rl/types.py` line by line - are all types captured?
- [ ] Verify Env behaviour methods are complete
- [ ] Check EnvGroupBuilder - is `compute_group_rewards` optional?
- [ ] Verify Trajectory/TrajectoryGroup methods
- [ ] Read `rl/rollouts.py` - are the function signatures correct?
- [ ] Read `rl/data_processing.py` - verify advantage formula
- [ ] Read `rl/train.py` - what's the minimal subset needed?

**Questions to answer:**
- What loss functions does Tinkex support? (importance_sampling, ppo, others?)
- Is async training included alongside sync in the Phase 2 plan?
- What's the actual flow of `train_step` vs `do_sync_training`?

### 4. DPO Analysis

- [ ] Verify DPO loss formula is correct
- [ ] Check how reference model is created and used
- [ ] Verify the chosen/rejected interleaving pattern
- [ ] Check if Tinkex has `forward_backward_custom`

### 5. Distillation Analysis

- [ ] Verify distillation types are accurate
- [ ] Check the teacher sampling flow
- [ ] Are there special loss functions for distillation?

### 6. Existing Elixir Code Review

- [ ] Check `supervised/train.ex` - what patterns can we reuse?
- [ ] Check `supervised/dataset.ex` - applicable to RL?
- [ ] Check `renderers/llama3.ex` - what's the actual pattern?
- [ ] Check `types/` - are there types we can reuse for RL?
- [ ] Check `eval/tinkex_generate.ex` - similar to completers?

### 7. Dependencies and Tinkex API

- [ ] Verify Tinkex has all required methods:
  - `forward_backward` with loss_fn parameter
  - `forward_backward_custom` for DPO
  - `optim_step` with AdamParams
  - `save_weights_for_sampler`
  - `save_state` / `load_state`
  - `compute_logprobs` for reference model
- [ ] Check ChzEx patterns for config structs

---

## Specific Verification Tasks

### Task 1: Renderer LOC Audit
```bash
# Run these and update the document with actual counts
wc -l ./tinker-cookbook/tinker_cookbook/renderers.py
grep -n "class.*Renderer" ./tinker-cookbook/tinker_cookbook/renderers.py
```

### Task 2: RL Types Completeness
```bash
# Extract all dataclass/class definitions from rl/types.py
grep -E "^(class|@dataclass)" ./tinker-cookbook/tinker_cookbook/rl/types.py
```

### Task 3: Tinkex API Surface
```bash
# Check what methods exist in Tinkex
grep -r "def " ./path/to/tinkex/lib/ | grep -E "(forward_backward|optim_step|save_|load_)"
```

### Task 4: Existing Elixir Patterns
```bash
# Check existing behaviour definitions
grep -r "@callback" ./lib/tinkex_cookbook/
```

---

## Document Updates

When you find errors or gaps, **edit the document directly**:

**File to edit:** `./docs/20251224/phase2_prerequisites/PHASE2_PART1_INFRASTRUCTURE.md`

For each correction:
1. Locate the incorrect section
2. Make the correction inline
3. Add a note if the change is significant: `<!-- REVIEW: Updated X based on Y -->`

---

## Output Format

After your review, append a section to the document:

```markdown
---

## Review Notes (2025-12-24)

### Corrections Made

1. **Section X.Y:** [What was wrong] → [What it should be]
2. ...

### Gaps Identified

1. **Missing component:** [Description]
2. ...

### Recommendations

1. [Recommendation]
2. ...

### Verified Accurate

- [X] Section 1.1 Renderers
- [ ] Section 2.1 RL Types (corrected)
- ...
```

---

## Execution

Run this review from the `tinkex_cookbook` root directory:

```bash
cd /home/home/p/g/North-Shore-AI/tinkex_cookbook

# The agent should:
# 1. Read the infrastructure document
# 2. Cross-reference with Python sources
# 3. Cross-reference with existing Elixir code
# 4. Edit the document with corrections
# 5. Append review notes
```

---

## Critical Questions to Answer

Before completing the review, ensure these are addressed in the document:

1. **What's the minimal infrastructure for `rl_basic`?**
   - The simplest RL recipe - what does it actually need?

2. **What's shared vs recipe-specific?**
   - Some infrastructure is shared, some is per-recipe. Is the split correct?

3. **Are the async patterns appropriate for Elixir?**
   - The plan requires sync + async together. Is the abstraction shared and correct?

4. **What about testing infrastructure?**
   - Mock tokenizers, mock clients - do we have these? Do we need more?

5. **Parity testing approach?**
   - How will we verify each component matches Python?

---

## Success Criteria

The review is complete when:

1. All LOC counts are verified against actual source
2. All type signatures are verified against Python code
3. All Elixir patterns are verified against existing code
4. Gaps and missing components are identified
5. The document is updated with corrections
6. Review notes are appended to the document
