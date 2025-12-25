#!/usr/bin/env python3
"""
Generate HuggingFace parity fixtures for Elixir renderer tests.

Run once to generate test/fixtures/hf_parity.json, then commit the file.
Elixir tests compare renderer output against these known-correct tokens.

Usage:
    python scripts/generate_hf_parity_fixtures.py

Requirements:
    pip install transformers torch
"""

import json
from datetime import date
from transformers import AutoTokenizer


# Models to test - using ungated models where possible
MODELS = [
    # Ungated models (no HF auth required)
    "Qwen/Qwen3-0.6B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",  # DeepSeek variant

    # Gated models (require HF_TOKEN with access)
    # "meta-llama/Llama-3.2-1B-Instruct",
    # "openai/gpt-oss-20b",
    # "moonshotai/Kimi-K2-Thinking",
]

# Test conversations
GENERATION_CONVO = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm fine, thank you!"},
    {"role": "user", "content": "What is the capital of France?"},
]

SUPERVISED_CONVO = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm fine, thank you!"},
]


def load_tokenizer(model_name: str):
    """Load tokenizer with special handling for some models."""
    kwargs = {}
    if model_name == "moonshotai/Kimi-K2-Thinking":
        kwargs["trust_remote_code"] = True
        kwargs["revision"] = "612681931a8c906ddb349f8ad0f582cb552189cd"
    return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)


def augment_convo_for_model(model_name: str, convo: list) -> list:
    """Add model-specific system messages."""
    if model_name.startswith("meta-llama"):
        today = date.today().strftime("%d %b %Y")
        system_msg = {
            "role": "system",
            "content": f"Cutting Knowledge Date: December 2023\nToday Date: {today}\n\n",
        }
        return [system_msg] + convo
    return convo


def generate_fixtures():
    """Generate all fixtures."""
    fixtures = {}

    for model_name in MODELS:
        print(f"Processing {model_name}...")
        try:
            tokenizer = load_tokenizer(model_name)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        model_fixtures = {
            "model_name": model_name,
            "generation": {},
            "supervised": {},
        }

        # Generation test (3-turn)
        gen_convo = augment_convo_for_model(model_name, GENERATION_CONVO)
        gen_tokens = tokenizer.apply_chat_template(
            GENERATION_CONVO,  # Use original for HF (it adds system internally for some)
            add_generation_prompt=True,
            tokenize=True,
        )
        model_fixtures["generation"] = {
            "messages": GENERATION_CONVO,
            "augmented_messages": gen_convo,
            "expected_tokens": gen_tokens,
            "decoded": tokenizer.decode(gen_tokens),
        }

        # Supervised test (2-turn)
        sup_convo = augment_convo_for_model(model_name, SUPERVISED_CONVO)

        # For Qwen3, HF includes thinking tags in supervised
        if model_name.startswith("Qwen"):
            sup_convo_hf = SUPERVISED_CONVO.copy()
            sup_convo_hf[1] = {
                "role": "assistant",
                "content": "<think>\n\n</think>\n\n I'm fine, thank you!"
            }
        else:
            sup_convo_hf = SUPERVISED_CONVO

        sup_tokens = tokenizer.apply_chat_template(
            sup_convo_hf,
            add_generation_prompt=False,
            tokenize=True,
        )
        model_fixtures["supervised"] = {
            "messages": SUPERVISED_CONVO,
            "augmented_messages": sup_convo,
            "expected_tokens": sup_tokens,
            "decoded": tokenizer.decode(sup_tokens),
        }

        # Normalize model name for JSON key (replace / with __)
        key = model_name.replace("/", "__")
        fixtures[key] = model_fixtures
        print(f"  OK: gen={len(gen_tokens)} tokens, sup={len(sup_tokens)} tokens")

    return fixtures


def main():
    fixtures = generate_fixtures()

    output_path = "test/fixtures/hf_parity.json"
    with open(output_path, "w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"\nWritten to {output_path}")
    print(f"Models: {len(fixtures)}")


if __name__ == "__main__":
    main()
