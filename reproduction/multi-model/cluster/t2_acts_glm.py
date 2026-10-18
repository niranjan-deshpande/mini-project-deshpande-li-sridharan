"""T2 step 1 for GLM-4.5-Air: score every passage of real rollouts with the DoM probes, in the context the model saw.

GLM's chat template shows an assistant turn's reasoning only if it comes after the last user message. Harness nudges
("Please proceed to the next step...") are user messages, so a rollout is cut into segments at each nudge: steps in
segment s were generated with earlier segments' reasoning hidden and segment s's own reasoning visible, which is
exactly render(messages up to the end of segment s). One forward pass per segment; passages of that segment's steps
are read from it. Forward hooks project every layer's output onto the DoM vectors; a passage's score is the max (and
mean) of <v, x_t> over its tokens.

Regions inside the step-th assistant turn: reasoning = inside <think>...</think>, content = from </think> to the
first <tool_call>, tool_calls = from the first <tool_call> to the end of the turn.

Usage (athena02): python t2_acts_glm.py --model <snapshot> --dom dom/glm/dom_vectors.npz --rollouts out_glm_full.jsonl --out t2/glm
"""
import argparse, json, os, re, time
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
ap.add_argument("--check", type=int, default=0, help="only render the first N rollouts and dump renders (no model)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
si, sn = map(int, a.shard.split("/"))

tok = AutoTokenizer.from_pretrained(a.model)
MARKERS = ("<|user|>", "<|observation|>", "<|system|>", "<|assistant|>")


def to_chat(chat):
    msgs = []
    for m in chat:
        if m["role"] == "assistant":
            x = {"role": "assistant", "content": m["content"] or ""}
            if m.get("reasoning"):
                x["reasoning_content"] = m["reasoning"]
            if m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    args = c["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}
                    calls.append({"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": args}})
                x["tool_calls"] = calls
            msgs.append(x)
        elif m["role"] == "tool":
            msgs.append({"role": "tool", "tool_call_id": m.get("tool_call_id"), "content": m["content"] or ""})
        else:
            msgs.append({"role": m["role"], "content": m["content"] or ""})
    return msgs


def segments(msgs):
    """[(end_index_exclusive, [assistant step numbers in segment])]; a new segment starts at every user message
    after the first one."""
    out, steps, step = [], [], 0
    seen_user = False
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            if seen_user and steps:
                out.append((i, steps)); steps = []
            seen_user = True
        elif m["role"] == "assistant":
            step += 1; steps.append(step)
    if steps:
        out.append((len(msgs), steps))
    return out


def regions(text, steps):
    """char regions per (step, channel) for the assistant turns of the rendered prefix; `steps` are the global
    step numbers whose reasoning is visible in this render (the last segment)."""
    starts = [m.start() for m in re.finditer(re.escape("<|assistant|>"), text)]
    out = {}
    for step in steps:
        if step - 1 >= len(starts):
            break
        t0 = starts[step - 1] + len("<|assistant|>")
        nxt = [x for x in (text.find(t, t0) for t in MARKERS) if x >= 0]
        t1 = min(nxt) if nxt else len(text)
        o, c = text.find("<think>", t0, t1), text.find("</think>", t0, t1)
        body = t0
        if o >= 0 and c >= 0:
            if c > o + len("<think>"):
                out[(step, "reasoning")] = (o + len("<think>"), c)
            body = c + len("</think>")
        tc = text.find("<tool_call>", body, t1)
        out[(step, "content")] = (body, tc if tc >= 0 else t1)
        if tc >= 0:
            out[(step, "tool_calls")] = (tc, t1)
    return out


rows = [json.loads(l) for i, l in enumerate(open(a.rollouts)) if i % sn == si]
rows = [r for r in rows if r["passages"]]

if a.check:
    for r in rows[: a.check]:
        msgs = to_chat(r["chat"])
        segs = segments(msgs)
        for end, steps in segs:
            text = tok.apply_chat_template(msgs[:end], tools=r.get("tools") or None, tokenize=False, add_generation_prompt=False)
            reg = regions(text, steps)
            have = {(int(p["step"]), p["channel"]) for p in r["passages"] if int(p["step"]) in steps}
            miss = have - set(reg)
            print(r["uid"], "segment end", end, "steps", steps[0], "-", steps[-1], "regions", len(reg), "passages", len(have), "missing", sorted(miss)[:5])
            for (st, ch), (s0, s1) in list(reg.items())[:3]:
                print("   ", st, ch, repr(text[s0:s0 + 80]))
    raise SystemExit

model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto",
                                             experts_implementation="grouped_mm").eval()
lm = model.model
DEV = model.get_input_embeddings().weight.device
keys = a.keys.split(",")
D = np.load(a.dom)
Vcpu = torch.tensor(np.stack([D[k] for k in keys], -1), dtype=torch.float32)  # [layers, d, keys]
NL = len(lm.layers)
Vdev = {}
proj = [None] * NL


def mk_hook(L):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if (L, h.device) not in Vdev:
            Vdev[(L, h.device)] = Vcpu[L].to(h.device)
        proj[L] = (h[0].float() @ Vdev[(L, h.device)]).cpu()  # [seq, keys]
    return hook


for L, layer in enumerate(lm.layers):
    layer.register_forward_hook(mk_hook(L))

MX, MN, index, skipped = [], [], [], []
t0 = time.time()
n_fw = 0
with torch.no_grad():
    for j, r in enumerate(rows):
        msgs = to_chat(r["chat"])
        want = {}
        for p in r["passages"]:
            want.setdefault(int(p["step"]), []).append(p)
        for end, steps in segments(msgs):
            todo = [p for st in steps for p in want.get(st, [])]
            if not todo:
                continue
            text = tok.apply_chat_template(msgs[:end], tools=r.get("tools") or None, tokenize=False, add_generation_prompt=False)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            ids, off = enc["input_ids"], np.array(enc["offset_mapping"])
            if len(ids) > a.max_tokens:
                skipped += [{"uid": r["uid"], "n": p["n"], "why": f"{len(ids)} tokens"} for p in todo]; continue
            reg = regions(text, steps)
            model(torch.tensor([ids], device=DEV), use_cache=False, logits_to_keep=1)
            n_fw += 1
            P = torch.stack(proj, 1).numpy()  # [seq, layers, keys]
            for p in todo:
                g = reg.get((int(p["step"]), p["channel"]))
                if g is None:
                    skipped.append({"uid": r["uid"], "n": p["n"], "why": "region not found"}); continue
                s0, s1 = g
                while s0 < s1 and text[s0].isspace(): s0 += 1
                while s1 > s0 and text[s1 - 1].isspace(): s1 -= 1
                t = np.where((off[:, 1] > s0) & (off[:, 0] < s1))[0]
                if len(t) == 0:
                    skipped.append({"uid": r["uid"], "n": p["n"], "why": "no tokens"}); continue
                MX.append(P[t].max(0)); MN.append(P[t].mean(0))
                index.append({"uid": r["uid"], "n": p["n"], "step": p["step"], "channel": p["channel"],
                              "tok0": int(t[0]), "tok1": int(t[-1]) + 1, "argmax": (t[P[t].argmax(0)]).tolist()})
        if j % 10 == 0:
            print(f"  {j + 1}/{len(rows)} rollouts, {n_fw} forwards, {len(ids)} tokens, {time.time() - t0:.0f}s", flush=True)

np.savez(f"{a.out}/scores_{si}.npz", max=np.stack(MX).astype(np.float32), mean=np.stack(MN).astype(np.float32),
         keys=np.array(keys))
with open(f"{a.out}/index_{si}.jsonl", "w") as f:
    for x in index:
        f.write(json.dumps(x) + "\n")
json.dump(skipped, open(f"{a.out}/skipped_{si}.json", "w"), indent=1)
print(f"DONE {len(index)} passages, {len(skipped)} skipped, {n_fw} forwards, {time.time() - t0:.0f}s")
