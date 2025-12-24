# Torch & Transformers Usage Analysis - December 20, 2025

## Overview

This directory contains the **definitive analysis** of how torch and transformers are actually used in tinker-cookbook, with complete Elixir/Nx equivalents for tinkex_cookbook.

**Critical Finding**: The cookbook is a **CLIENT library** that calls the Tinker API for training. It does NOT implement model training itself. torch/transformers are used only for data preparation, NOT for ML frameworks.

---

## Files in This Analysis

### 1. 12_torch_transformers_actual_usage.md (23KB)
**The main comprehensive report.**

Contains:
- Complete analysis of all 12 files importing torch
- Complete analysis of all 3 files importing transformers
- Categorization: Tokenization (100% solved), Model Metadata (optional), Tensor Ops (trivial Nx), API Bridge (need TensorData)
- Detailed module-by-module breakdown with Elixir equivalents
- Implementation plan (4 phases, 2-3 weeks)
- Architecture diagrams showing client/server separation
- Appendix with complete file-by-file usage table

**Read this for**: Complete understanding of the porting task

---

### 2. 12_torch_transformers_actual_usage_SUMMARY.txt (3.7KB)
**Executive summary for quick reference.**

Contains:
- Key findings (4 categories)
- Dependencies needed (Nx + tokenizers)
- Implementation plan checklist
- Clear "NO" answer to "Do we need torch/transformers wrappers?"

**Read this for**: Quick briefing or management summary

---

### 3. 12a_porting_cheatsheet.md (6.6KB)
**Practical porting guide for developers.**

Contains:
- Side-by-side Python/Elixir code examples
- Tensor creation, operations, reductions, math functions
- 5 real cookbook patterns with translations:
  1. Compute advantages (RL)
  2. Weighted dot product (loss computation)
  3. Create training datum
  4. DPO loss computation
  5. Concatenate token chunks (renderers)
- Async patterns, error handling, logging
- List of PyTorch features NOT used (torch.nn, torch.optim, autograd)

**Read this for**: Actual coding/porting work

---

### 4. 12b_tensor_data_implementation.ex (11KB)
**Complete, ready-to-use implementation of Tinkex.TensorData.**

Contains:
- Full module implementation with docs and tests
- `from_nx/1`, `to_nx/1`, `to_map/1`, `from_map/1` functions
- Complete dtype mapping (16 types: float16/32/64, int8/16/32/64, etc.)
- Usage examples in Datum construction
- Side-by-side Python vs Elixir comparison
- ExUnit test suite

**Read this for**: Drop-in implementation (1-2 day task)

---

## Quick Start Guide

### For Decision Makers

1. Read: `12_torch_transformers_actual_usage_SUMMARY.txt`
2. Key takeaway: **NO** torch/transformers wrappers needed. Just Nx + business logic porting.
3. Effort: 2-3 weeks for full feature parity

### For Developers

1. Read: `12a_porting_cheatsheet.md` (familiarize with Nx equivalents)
2. Implement: Copy `12b_tensor_data_implementation.ex` to `tinkex/lib/tinkex/tensor_data.ex`
3. Add dependency: `{:nx, "~> 0.9"}` to tinkex/mix.exs
4. Port modules: Use cheatsheet patterns to convert cookbook modules

### For Architects

1. Read: `12_torch_transformers_actual_usage.md` (complete analysis)
2. Focus on: "Training Architecture" section (client/server separation)
3. Reference: Implementation plan (Phases 1-4)

---

## Key Insights

### 1. No Model Training Code
```
tinker-cookbook (CLIENT)          Tinker API (SERVER)
├─ Tokenize data                  ├─ Hosts PyTorch models
├─ Prepare tensors                ├─ Implements backprop
├─ Create Datum payloads          ├─ Manages LoRA adapters
├─ HTTP POST to API               ├─ Runs optimizers
└─ Receive loss/gradients         └─ Returns logprobs/loss
```

The cookbook NEVER implements `model.forward()`, `loss.backward()`, or `optimizer.step()`. All training happens server-side.

### 2. Torch Usage is Minimal
**Used**: tensor creation, concatenation, arithmetic (mean, sum, dot)
**NOT used**: torch.nn, torch.optim, torch.autograd, GPU ops

All used operations have trivial Nx equivalents:
- `torch.tensor([1,2,3])` → `Nx.tensor([1,2,3])`
- `torch.cat([t1, t2])` → `Nx.concatenate([t1, t2])`
- `tensor.mean()` → `Nx.mean(tensor)`

### 3. Transformers is Only for Tokenization
**Used**: `AutoTokenizer.from_pretrained()`, `tokenizer.encode()`
**Already solved**: `{:tokenizers, "~> 0.5"}` in tinkex (HuggingFace tokenizers bindings)

Optional: `AutoConfig.from_pretrained()` for model metadata (can hardcode or skip)

### 4. The Missing Piece: TensorData Bridge
The ONLY new code needed is `Tinkex.TensorData` to convert between:
- Nx.Tensor ↔ JSON map for API payloads
- Python: `tinker.TensorData.from_torch(tensor)`
- Elixir: `Tinkex.TensorData.from_nx(tensor)`

See `12b_tensor_data_implementation.ex` for complete implementation.

---

## Dependencies Summary

### Required
```elixir
# tinkex/mix.exs
{:nx, "~> 0.9"}           # NEW - tensor operations
{:tokenizers, "~> 0.5"}   # Already have - HF tokenizers
{:req, "~> 0.5"}          # Already have - HTTP client
{:jason, "~> 1.4"}        # Already have - JSON
```

### NOT Required
- `torch` (Python package) - Nx replaces it
- `transformers` (Python package) - Tokenizers replaces it
- `axon` (Elixir package) - We don't do local training

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (1-2 days)
- [ ] Add `{:nx, "~> 0.9"}` to tinkex/mix.exs
- [ ] Implement `Tinkex.TensorData` (copy from 12b file)
- [ ] Add tests for TensorData roundtrips

### Phase 2: Renderers (3-5 days)
- [ ] Port `RoleColonRenderer`
- [ ] Port `Llama3Renderer`
- [ ] Port `Qwen3Renderer` (3 variants)
- [ ] Port `DeepSeekV3Renderer` (2 variants)
- [ ] Port `GptOssRenderer`

### Phase 3: Training Orchestration (5-7 days)
- [ ] Port `supervised/train.py` (main training loop)
- [ ] Port `supervised/common.py` (helpers)
- [ ] Port `preference/train_dpo.py` (DPO training)
- [ ] Port checkpoint management

### Phase 4: RL Support (3-5 days)
- [ ] Port `rl/data_processing.py` (trajectory handling)
- [ ] Port `rl/metrics.py` (KL divergence, etc.)
- [ ] Port `rl/train.py` (RL orchestration)

**Total**: 12-19 days (~2-3 weeks)

---

## Related Documents

- **Supersedes**: `08_torch_transformers_axon_mapping.md` (incorrectly assumed model training porting)
- **Complements**: API documentation in tinkex project
- **References**: Python tinker-cookbook source code

---

## Verification Checklist

When implementing, verify that:
- [ ] NO torch or transformers dependencies in mix.exs
- [ ] All tensor ops use Nx, not custom bindings
- [ ] TensorData correctly serializes/deserializes to/from API JSON
- [ ] Renderers produce identical token sequences to Python versions
- [ ] Training loops correctly call Tinkex.TrainingClient API methods
- [ ] No local model weights, optimizers, or gradient computation

---

## Questions & Answers

**Q: Do we need Axon for tinkex_cookbook?**
A: NO. Axon is for defining and training neural networks locally. The cookbook never does this - all training is server-side via Tinker API.

**Q: Do we need PyTorch bindings?**
A: NO. The cookbook uses torch only for basic tensor ops (create, concat, arithmetic), which Nx handles natively.

**Q: Do we need Transformers bindings?**
A: NO. We only need tokenization, which the `tokenizers` library already provides.

**Q: What's the hardest part of the port?**
A: Porting the business logic in renderers and training orchestration. The tensor ops themselves are trivial.

**Q: Can we start using tinkex_cookbook before all renderers are done?**
A: YES. Start with one renderer (e.g., Llama3), implement supervised training, then add more renderers/features incrementally.

---

## Contact

For questions about this analysis:
- See detailed code examples in files 12a (cheatsheet) and 12b (implementation)
- Check Python source: `tinkerer/thinking-machines-labs/tinker-cookbook/`
- Reference Nx docs: https://hexdocs.pm/nx/

**Last Updated**: December 20, 2025
**Analyzer**: Claude Opus 4.5 (via North-Shore-AI Claude Code)
