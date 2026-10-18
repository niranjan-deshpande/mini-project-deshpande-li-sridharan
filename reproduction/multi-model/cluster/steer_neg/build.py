"""Builds the cut-point prompts and steering vectors. For every GLM impossible-split rollout with an enacted hack
(unanimous judge label), cut just before the first assistant turn that enacts it and render the prefix exactly as
the model saw it (same renderer as t2_acts_glm.py, validated token-identical to vLLM), plus the generation prompt.
Vectors: span__all at layer 33, scaled; random direction seeded as in t3_steer.py (norm-matched to v)."""
import collections, json, sys
import numpy as np, torch
from transformers import AutoTokenizer
src = open("/data/agastyas/rh_probe/t2_acts_glm.py").read()
exec(src[src.index("def to_chat"):src.index("def segments")])

import glob
SNAP = glob.glob("/data/hf_cache/models--zai-org--GLM-4.5-Air-FP8/snapshots/*")[0]
tok = AutoTokenizer.from_pretrained(SNAP)
labels = collections.defaultdict(list)
for l in open("steer_neg/labels_glm.jsonl"):
    p = json.loads(l); labels[p["uid"]].append(p)
out = open("steer_neg/prompts.jsonl", "w")
lens = []
for l in open("out_glm_imp.jsonl"):
    r = json.loads(l)
    ps = labels.get(r["uid"])
    if not ps or ps[0]["split"] not in ("conflicting", "oneoff"):
        continue
    en = [p for p in ps if p["gt"] and "enacted" in p["judge_statuses"]]
    if not en:
        continue
    s = min(p["step"] for p in en)
    msgs = to_chat(r["chat"])
    k = [i for i, m in enumerate(msgs) if m["role"] == "assistant"][s - 1]
    text = tok.apply_chat_template(msgs[:k], tools=r.get("tools") or None, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    orig = r["chat"][[i for i, m in enumerate(r["chat"]) if m["role"] == "assistant"][s - 1]]
    lens.append(len(ids))
    out.write(json.dumps({"uid": r["uid"], "split": r["split"], "step": s, "ids": ids, "tail": text[-12000:],
                          "orig_channels": sorted({p["channel"] for p in en if p["step"] == s}),
                          "orig_categories": sorted({c for p in en if p["step"] == s for c in p["judge_categories"]}),
                          "orig_turn": {"reasoning": orig.get("reasoning"), "content": orig.get("content"),
                                        "tool_calls": orig.get("tool_calls")}}) + "\n")
print("prompts", len(lens), "tokens median", int(np.median(lens)), "max", max(lens), "ends with", repr(text[-40:]))

D = np.load("dom/glm/dom_vectors.npz")
v = torch.tensor(D["span__all"][33], dtype=torch.float32)
g = torch.Generator().manual_seed(1234)
r = torch.randn(v.shape, generator=g); r = r / r.norm() * v.norm()
for name, vec in (("neg2", -2 * v), ("neg3", -3 * v), ("rand3", 3 * r)):
    np.save(f"steer_neg/{name}.npy", vec.numpy())
print("|v|", float(v.norm()))
