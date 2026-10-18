"""Per-token probe projections for single-turn gpt-oss samples (Phase 0 calibration and RL evaluation).

Each sample is rendered the way `vllm serve` rendered it (vLLM's harmony preamble with the sample's reasoning
effort, the user message, then the assistant's analysis message and, if the sample finished, its final message),
and one forward pass of gpt-oss-120b records <v, h_t> for every DoM key at every layer, for every assistant token.
Saved per sample: float32 projections [tokens, layers, keys] (fp16 overflows: gpt-oss projections exceed 65504) for the analysis + final region and the char offsets
of each token within reasoning / content, so any layer, key and aggregation (max, paragraph max) can be calibrated
offline.

The sampler stores a completion as reasoning + "</think>" + content (see sample.py --chat).

Usage (athena02): python score_probe_gptoss.py SAMPLES.jsonl TASKS.jsonl OUT_DIR --shard i/n
"""
import argparse, glob, json, os, time
import numpy as np, torch
from openai_harmony import Message, Role
from vllm.entrypoints.openai.parser.harmony_utils import build_harmony_preamble, get_encoding
from env_lcb import user_prompt

ap = argparse.ArgumentParser()
ap.add_argument("samples"); ap.add_argument("tasks"); ap.add_argument("out")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--model", default=glob.glob("/data/agastyas/hf_cache_latekey/models--openai--gpt-oss-120b/snapshots/*")[0])
ap.add_argument("--dom", default="/data/agastyas/rh_probe/dom/gptoss_h/dom_vectors.npz")
ap.add_argument("--keys", default="span__all,turn__all,last__all")
ap.add_argument("--attn", default="kernels-community/vllm-flash-attn3")
ap.add_argument("--check-only", action="store_true")
a = ap.parse_args()
si, sn = map(int, a.shard.split("/"))
os.makedirs(f"{a.out}/arr", exist_ok=True)
enc = get_encoding()
MSG_TOK = enc.encode("<|message|>", allowed_special="all")[0]


def asst(channel, text):
    return Message.from_role_and_content(Role.ASSISTANT, text).with_channel(channel)


def render(task, mode, effort, comp):
    """Token ids plus, for the reasoning and the content, (first token, char offsets of each token in that text)."""
    reasoning, content = comp.split("</think>", 1) if "</think>" in comp else (comp, "")
    msgs = build_harmony_preamble(instructions=None, tools=None, reasoning_effort=effort, with_custom_tools=False)
    msgs.append(Message.from_role_and_content(Role.USER, user_prompt(task, mode)))
    ids = []
    for m in msgs:
        ids += enc.render(m)
    parts = []
    for ch, text in [("analysis", reasoning), ("final", content)]:
        if not text:
            continue
        t = enc.render(asst(ch, text))
        k = t.index(MSG_TOK) + 1
        body = t[k:-1]  # message text tokens (drop trailing <|end|>)
        offs, pos = [], 0
        for tid in body:
            s = enc.decode([tid]); offs.append((pos, pos + len(s))); pos += len(s)
        parts.append({"channel": ch, "tok0": len(ids) + k, "n": len(body), "offsets": offs,
                      "ok": enc.decode(body) == text})
        ids += t
    return ids, parts


tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(a.tasks)}
rows = [json.loads(l) for i, l in enumerate(open(a.samples)) if i % sn == si]

if a.check_only:
    bad = tot = 0
    for r in rows[:20]:
        for comp in r["completions"]:
            ids, parts = render(tasks[r["task_id"]], r["mode"], r["effort"], comp)
            tot += 1; bad += not all(p["ok"] for p in parts)
    print(f"{bad}/{tot} samples with text/token mismatch")
    raise SystemExit

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0",
                                             attn_implementation=a.attn).eval()
keys = a.keys.split(",")
D = np.load(a.dom)
V = torch.tensor(np.stack([D[k] for k in keys], -1), dtype=torch.float32, device="cuda:0")  # [layers, d, keys]
proj = [None] * V.shape[0]


def mk_hook(L):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        proj[L] = (h[0].float() @ V[L]).cpu()
    return hook


for L, layer in enumerate(model.model.layers):
    layer.register_forward_hook(mk_hook(L))

outp = f"{a.out}/scores_{si}.jsonl"
done = {(json.loads(l)["task_id"], json.loads(l)["k"]) for l in open(outp)} if os.path.exists(outp) else set()
t0 = time.time()
with open(outp, "a") as f, torch.no_grad():
    for j, r in enumerate(rows):
        for k, comp in enumerate(r["completions"]):
            if (r["task_id"], k) in done:
                continue
            ids, parts = render(tasks[r["task_id"]], r["mode"], r["effort"], comp)
            model(torch.tensor([ids], device="cuda:0"), use_cache=False, logits_to_keep=1)
            P = torch.stack(proj, 1)  # [tokens, layers, keys]
            name = f"{r['task_id']}__{r['effort']}__{k}"
            save, meta = {}, []
            for p in parts:
                save[f"proj_{p['channel']}"] = P[p["tok0"]:p["tok0"] + p["n"]].numpy().astype(np.float32)
                save[f"offs_{p['channel']}"] = np.array(p["offsets"], dtype=np.int32).reshape(-1, 2)
                meta.append({"channel": p["channel"], "n": p["n"], "ok": p["ok"]})
            np.savez_compressed(f"{a.out}/arr/{name}.npz", **save)
            f.write(json.dumps({"task_id": r["task_id"], "effort": r["effort"], "k": k, "n_ctx": len(ids),
                                "parts": meta, "keys": keys, "arr": name}) + "\n"); f.flush()
        if j % 10 == 0:
            print(f"  {j + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
print("DONE")
