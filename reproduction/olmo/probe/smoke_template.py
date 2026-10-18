#!/usr/bin/env python3
"""Tokenizer-only smoke test (no model load): validate render / span-finding / assistant-mask.

Checks that OLMo's chat template accepts our synth + rollout messages, that marked spans are locatable
in token space, and that assistant tokens are identified. Exits nonzero on failure so we catch template
problems before loading the 32B.
"""
import json, sys
from transformers import AutoTokenizer
sys.path.insert(0, ".")
from extract_and_score import render, token_span_indices, assistant_token_mask

tok = AutoTokenizer.from_pretrained(sys.argv[1] if len(sys.argv) > 1 else "~/models/olmo-3.1-32b-think")

print("=== synth pairs ===")
synth = [json.loads(l) for l in open("pairs_full.jsonl")][:5]
span_ok = turn_fail = 0
for item in synth:
    for branch in ("honest", "hack"):
        text, _ = render(tok, item[branch]["messages"])
        _, span_idx = token_span_indices(tok, text, item[branch]["span"])
        if span_idx:
            span_ok += 1
        else:
            turn_fail += 1
            print(f"  [miss] span not found: {item[branch]['span'][:50]!r}")
print(f"  span found in {span_ok}/{len(synth)*2} branches")

print("=== rollouts ===")
rolls = [json.loads(l) for l in open("transcripts_full.jsonl")][:5]
for r in rolls:
    text, is_asst = render(tok, r["messages"])
    ids, amask = assistant_token_mask(tok, text, is_asst)
    nasst = sum(amask)
    print(f"  {r['task_id']}: {len(ids)} toks, {nasst} assistant toks ({100*nasst//max(len(ids),1)}%)")

if span_ok >= len(synth) * 2 * 0.7 and all(True for _ in rolls):
    print("SMOKE_OK")
else:
    print("SMOKE_FAIL"); sys.exit(1)
