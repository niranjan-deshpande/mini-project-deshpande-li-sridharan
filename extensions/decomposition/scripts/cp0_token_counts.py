"""CHECKPOINT 0: rollout token counts with the Qwen chat template (rendered exactly as t2_acts.py does),
assistant-token counts, and the layer-40 fp16 cache size."""
import gzip, json, sys, numpy as np
from transformers import AutoTokenizer

ROLL = "../../reproduction/multi-model/results/rollouts/qwen_rollouts.jsonl.gz"
tok = AutoTokenizer.from_pretrained("data/qwen_tok")


def to_chat(r):  # copied from reproduction/multi-model/cluster/t2_acts.py
    msgs = []
    for m in r["chat"]:
        if m["role"] == "assistant":
            x = {"role": "assistant", "content": m["content"] or ""}
            if m.get("reasoning"):
                x["reasoning_content"] = m["reasoning"]
            if m.get("tool_calls"):
                x["tool_calls"] = [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": c["arguments"]}} for c in m["tool_calls"]]
            msgs.append(x)
        elif m["role"] == "tool":
            msgs.append({"role": "tool", "tool_call_id": m.get("tool_call_id"), "content": m["content"] or ""})
        else:
            msgs.append({"role": m["role"], "content": m["content"] or ""})
    return msgs


out = []
for line in gzip.open(ROLL, "rt"):
    r = json.loads(line)
    text = tok.apply_chat_template(to_chat(r), tools=r.get("tools") or None, tokenize=False,
                                   add_generation_prompt=False, preserve_thinking=True)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    off = np.array(enc["offset_mapping"])
    # assistant regions: from "<|im_start|>assistant" to the next "<|im_end|>"
    n_asst, pos = 0, 0
    while True:
        t0 = text.find("<|im_start|>assistant", pos)
        if t0 < 0:
            break
        t1 = text.find("<|im_end|>", t0); t1 = len(text) if t1 < 0 else t1
        n_asst += int(((off[:, 1] > t0) & (off[:, 0] < t1)).sum()); pos = t1
    out.append({"uid": r["uid"], "split": r["split"], "n_tokens": len(enc["input_ids"]), "n_asst": n_asst,
                "n_passages": len(r["passages"])})
    print(len(out), end="\r", file=sys.stderr)

json.dump(out, open("results/cp0/rollout_token_counts.json", "w"), indent=0)
T = np.array([o["n_tokens"] for o in out]); A = np.array([o["n_asst"] for o in out])
print(f"rollouts {len(T)}  total tokens {T.sum():,}  median {int(np.median(T)):,}  p90 {int(np.percentile(T,90)):,}  max {T.max():,}")
print(f"> 140k tokens: {(T > 140000).sum()}   > 100k: {(T > 100000).sum()}")
print(f"assistant tokens {A.sum():,} ({A.sum()/T.sum():.0%});  L40 fp16 cache {A.sum()*5120*2/1e9:.1f} GB (assistant only), "
      f"{T.sum()*5120*2/1e9:.1f} GB (all tokens)")
for s in ("conflicting", "oneoff", "original"):
    m = np.array([o["split"] == s for o in out])
    print(f"  {s:12} n={m.sum():3}  tokens {T[m].sum():,}  asst {A[m].sum():,}")
print("longest:", sorted(out, key=lambda o: -o["n_tokens"])[0])
