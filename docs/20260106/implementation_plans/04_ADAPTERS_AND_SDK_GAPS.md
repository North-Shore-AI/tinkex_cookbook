# Adapters and SDK Gaps Plan

Date: 2026-01-06

## Adapter Ownership Migration

### Tasks

- Create kitchen-owned noops:
  - CrucibleKitchen.Adapters.Noop.LLMClient
  - CrucibleKitchen.Adapters.Noop.EmbeddingClient
  - CrucibleKitchen.Adapters.Noop.VectorStore
- Update manifests to reference kitchen noops.
- Remove remaining adapter modules from tinkex_cookbook.

### Acceptance

- No adapter references to crucible_train noops.
- No adapter modules remain in tinkex_cookbook.

## LLM SDK Adapters

### claude_agent_sdk

- Implement LLMClient adapter that supports streaming and tool use.
- Map response to `content`, `usage`, `cost`, `raw` fields.
- Add retry policy for rate limits and transient errors.

### codex_sdk

- Implement LLMClient adapter with exec JSONL and app-server transports.
- Normalize tool output and file change artifacts.
- Track per-run costs and events.

### gemini_ex

- Implement LLMClient adapter with tool calling and system instruction support.
- Support structured output via response schema.
- Normalize error handling with other LLM adapters.

### Acceptance

- All LLMClient adapters return unified response shape.
- Provider-specific options are passed via config, not hardcoded.

## Vector and Embedding Adapters

- Implement EmbeddingClient adapters using gemini_ex or embed_ex.
- Implement VectorStore adapter for Chroma.
- Provide noop implementations for dev and tests.

## Tinkex SDK Gaps (Blocking)

### Required Endpoints

- Server capabilities endpoint.
- Health check endpoint.
- Model info endpoint.
- Typed response structs for training runs.
- forward_backward_custom or equivalent for DPO.

### Acceptance

- Kitchen adapters can call endpoints with typed responses.
- DPO recipe can execute custom loss path.

## Tokenizer Integration (tiktoken_ex)

- Define tokenizer adapter that aligns with renderers and adds no special tokens unless configured.
- Ensure BOS and EOS control matches parity requirements.
- Add deterministic tokenization tests.

## Python Bridge (snakebridge)

- Create manifests for `math_verify`, `sympy`, `pylatexenc`.
- Add adapter to kitchen with error wrapping and timeouts.
- Add deterministic tests for math verification on known problems.
