"""T2 step 1: score every passage of real rollouts with the DoM probes, using the full rollout as context.

Each rollout is rendered with the model's own chat template (served tool schemas, prior reasoning preserved, as vLLM
served it) and run through the model once. Forward hooks project every layer's output onto the DoM vectors; a
passage's score is the max (and mean) of <v, x_t> over its tokens. Passages are (step, channel) regions of the
step-th assistant turn: reasoning = inside <think>, content = between </think> and the first <tool_call>,
tool_calls = from the first <tool_call> to the end of the turn.

Usage (athena02): python t2_acts.py --model <snapshot> --dom dom/qwen/dom_vectors.npz --rollouts out/qwen_full.jsonl --out t2/qwen
Outputs: scores.npz (max, mean: [n_passages, layers, keys]), index.jsonl (uid, n per row), skipped.json
"""
import argparse, json, os, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dom", required=True)
ap.add_argument("--keys", default="span__all,turn__all,last__all")
ap.add_argument("--rollouts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--shard", default="0/1", help="i/n: process rollouts with index %% n == i")
ap.add_argument("--max-tokens", type=int, default=140000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
si, sn = map(int, a.shard.split("/"))

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
keys = a.keys.split(",")
D = np.load(a.dom)
V = torch.tensor(np.stack([D[k] for k in keys], -1), dtype=torch.float32, device="cuda:0")  # [layers, d, keys]
NL = V.shape[0]

proj = [None] * NL


def mk_hook(L):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        proj[L] = (h[0].float() @ V[L]).cpu()  # [seq, keys]
    return hook


for L, layer in enumerate(lm.layers):
    layer.register_forward_hook(mk_hook(L))


def to_chat(r):
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


def regions(text, n_steps):
    """char regions per (step, channel) for the assistant turns of the rendered conversation."""
    out, pos = {}, 0
    for step in range(1, n_steps + 1):
        t0 = text.find("<|im_start|>assistant", pos)
        if t0 < 0:
            break
        t0 += len("<|im_start|>assistant")
        t1 = text.find("<|im_end|>", t0)
        t1 = len(text) if t1 < 0 else t1
        pos = t1
        o, c = text.find("<think>", t0, t1), text.find("</think>", t0, t1)
        body = t0
        if o >= 0 and c >= 0:
            out[(step, "reasoning")] = (o + len("<think>"), c)
            body = c + len("</think>")
        tc = text.find("<tool_call>", body, t1)
        out[(step, "content")] = (body, tc if tc >= 0 else t1)
        if tc >= 0:
            out[(step, "tool_calls")] = (tc, t1)
    return out


rows = [json.loads(l) for i, l in enumerate(open(a.rollouts)) if i % sn == si]
rows = [r for r in rows if r["passages"]]
MX, MN, index, skipped = [], [], [], []
t0 = time.time()
with torch.no_grad():
    for j, r in enumerate(rows):
        text = tok.apply_chat_template(to_chat(r), tools=r.get("tools") or None, tokenize=False,
                                       add_generation_prompt=False, preserve_thinking=True)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, off = enc["input_ids"], np.array(enc["offset_mapping"])
        if len(ids) > a.max_tokens:
            skipped.append({"uid": r["uid"], "why": f"{len(ids)} tokens"}); continue
        reg = regions(text, r["n_steps"])
        model(torch.tensor([ids], device="cuda:0"), use_cache=False, logits_to_keep=1)
        P = torch.stack(proj, 1).numpy()  # [seq, layers, keys]
        for p in r["passages"]:
            g = reg.get((p["step"], p["channel"]))
            if g is None:
                skipped.append({"uid": r["uid"], "n": p["n"], "why": "region not found"}); continue
            s0, s1 = g
            # trim surrounding whitespace so boundary newlines are not counted
            while s0 < s1 and text[s0].isspace(): s0 += 1
            while s1 > s0 and text[s1 - 1].isspace(): s1 -= 1
            t = np.where((off[:, 1] > s0) & (off[:, 0] < s1))[0]
            if len(t) == 0:
                skipped.append({"uid": r["uid"], "n": p["n"], "why": "no tokens"}); continue
            MX.append(P[t].max(0)); MN.append(P[t].mean(0))
            index.append({"uid": r["uid"], "n": p["n"], "step": p["step"], "channel": p["channel"],
                          "tok0": int(t[0]), "tok1": int(t[-1]) + 1, "argmax": (t[P[t].argmax(0)]).tolist()})
        if j % 10 == 0:
            print(f"  {j + 1}/{len(rows)} rollouts, {len(ids)} tokens, {time.time() - t0:.0f}s", flush=True)

np.savez(f"{a.out}/scores_{si}.npz", max=np.stack(MX).astype(np.float32), mean=np.stack(MN).astype(np.float32),
         keys=np.array(keys))
with open(f"{a.out}/index_{si}.jsonl", "w") as f:
    for x in index:
        f.write(json.dumps(x) + "\n")
json.dump(skipped, open(f"{a.out}/skipped_{si}.json", "w"), indent=1)
print(f"DONE {len(index)} passages, {len(skipped)} skipped, {time.time() - t0:.0f}s")
