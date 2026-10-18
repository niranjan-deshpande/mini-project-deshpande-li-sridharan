"""Per-token probe projections for single-turn samples (Phase 0 calibration and RL evaluation).

For each sample: render prompt (same template, mode and effort as sampling) + completion text, one forward pass
of Qwen3.8-27B (optionally with a LoRA adapter), and record <v, h_t> at the steering-free residual output of the
probe layers for every completion token. Saved per sample: float16 projections [tokens, keys], the completion's
char offsets per token, and where </think> falls, so any aggregation (max, paragraph max, top-k mean) can be
calibrated offline.

Usage: python score_probe.py SAMPLES.jsonl TASKS.jsonl OUT_DIR --shard i/n [--adapter DIR]
"""
import argparse, glob, json, os, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from env_lcb import user_prompt

ap = argparse.ArgumentParser()
ap.add_argument("samples"); ap.add_argument("tasks"); ap.add_argument("out")
ap.add_argument("--shard", default="0/1"); ap.add_argument("--adapter", default=None)
ap.add_argument("--dom", default="/data/agastyas/rh_probe/dom/qwen/dom_vectors.npz")
ap.add_argument("--probes", default="turn__all:40,span__all:40,last__all:40")
a = ap.parse_args()
si, sn = map(int, a.shard.split("/"))
os.makedirs(a.out, exist_ok=True)
SNAP = glob.glob("/data/hf_cache/models--Qwen--Qwen3.8-27B/snapshots/*")[0]
tok = AutoTokenizer.from_pretrained(SNAP)
model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.bfloat16, device_map="cuda:0").eval()
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).eval()
base = model.get_base_model() if a.adapter else model
lm = base.model.language_model if hasattr(base.model, "language_model") else base.model
D = np.load(a.dom)
probes = [(k, int(L)) for k, L in (p.split(":") for p in a.probes.split(","))]
layers = sorted({L for _, L in probes})
V = {L: torch.tensor(np.stack([D[k][L] for k, LL in probes if LL == L], -1), dtype=torch.float32, device="cuda:0")
     for L in layers}
cap = {}


def mk(L):
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        cap[L] = (h[0].float() @ V[L]).cpu()
    return hook


for L in layers:
    lm.layers[L].register_forward_hook(mk(L))

tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(a.tasks)}
rows = [json.loads(l) for i, l in enumerate(open(a.samples)) if i % sn == si]
outp = f"{a.out}/scores_{si}.jsonl"
done = {(json.loads(l)["task_id"], json.loads(l)["k"]) for l in open(outp)} if os.path.exists(outp) else set()
arr_dir = f"{a.out}/arr"; os.makedirs(arr_dir, exist_ok=True)
t0 = time.time()
with open(outp, "a") as f, torch.no_grad():
    for j, r in enumerate(rows):
        t = tasks[r["task_id"]]
        p_ids = tok(tok.apply_chat_template([{"role": "user", "content": user_prompt(t, r["mode"])}], tokenize=False,
                                            add_generation_prompt=True, reasoning_effort=r["effort"]),
                  add_special_tokens=False)["input_ids"]
        for k, comp in enumerate(r["completions"]):
            if (r["task_id"], k) in done:
                continue
            enc = tok(comp, add_special_tokens=False, return_offsets_mapping=True)
            ids = list(p_ids) + enc["input_ids"]
            model(torch.tensor([ids], device="cuda:0"), use_cache=False, logits_to_keep=1)
            P = torch.cat([cap[L] for L in layers], -1)[len(p_ids):].numpy().astype(np.float16)
            name = f"{r['task_id']}__{k}".replace("/", "_")
            np.savez_compressed(f"{arr_dir}/{name}.npz", proj=P, offsets=np.array(enc["offset_mapping"], dtype=np.int32))
            f.write(json.dumps({"task_id": r["task_id"], "k": k, "n_tok": len(enc["input_ids"]),
                                "think_end": comp.find("</think>"), "keys": [f"{kk}:{L}" for kk, L in probes],
                                "max": P.max(0).astype(float).tolist(), "arr": name}) + "\n"); f.flush()
        if j % 10 == 0:
            print(f"  {j + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
print("DONE")
