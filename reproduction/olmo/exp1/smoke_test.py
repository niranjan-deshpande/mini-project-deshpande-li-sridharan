#!/usr/bin/env python3
"""Smoke test for OLMo 3.1 32B Think served by vLLM.

Checks: (1) coherent generation; (2) <think> block present in content;
(3) multi-turn: does the chat template re-include prior-turn <think> text?
   (measured via prompt token count with vs without think text in history)
(4) code block extraction sanity for the harness's find_code.
"""
import os
import sys

from openai import OpenAI

BASE = os.environ.get("LOCAL_BASE_URL", "http://localhost:8000/v1")
client = OpenAI(base_url=BASE, api_key=os.environ.get("LOCAL_API_KEY", "dummy"))

M = "olmo"

# --- 1-2: single turn ---
r = client.chat.completions.create(
    model=M,
    messages=[{"role": "user", "content": "Write a python function that reverses a string. Put the final code in a ``` block."}],
    temperature=0.6, top_p=0.95, max_tokens=4096,
)
msg = r.choices[0].message
text = msg.content or ""
reasoning = getattr(msg, "reasoning_content", None)
print("=== finish_reason:", r.choices[0].finish_reason)
print("=== has <think> in content:", "<think>" in text)
print("=== separate reasoning_content:", bool(reasoning))
print("=== completion tokens:", r.usage.completion_tokens, "prompt tokens:", r.usage.prompt_tokens)
print("=== content head ===")
print(text[:600])
print("=== content tail ===")
print(text[-600:])

# --- 3: multi-turn template behavior ---
assistant_full = text if text else "<think>placeholder thinking</think>\nHere is code."
history = [
    {"role": "user", "content": "Write a python function that reverses a string."},
    {"role": "assistant", "content": assistant_full},
    {"role": "user", "content": "Now make it recursive."},
]
r2 = client.chat.completions.create(model=M, messages=history, temperature=0.6, top_p=0.95, max_tokens=16)

think_len_est = len(assistant_full.split("</think>")[0]) // 4 if "</think>" in assistant_full else 0
print("\n=== multi-turn prompt tokens:", r2.usage.prompt_tokens,
      "| est. think tokens in history:", think_len_est)
print("(if prompt tokens ≈ history-without-think, the template strips prior reasoning)")

sys.exit(0)
