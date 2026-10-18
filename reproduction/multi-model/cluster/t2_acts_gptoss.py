"""T2 step 1 for gpt-oss: same as t2_acts.py, but the rollout is rendered with vLLM's own harmony code path (the
preamble, message parsing and per-message token rendering that `vllm serve` used), so passages map to harmony
messages exactly: reasoning -> analysis message, content -> commentary/final text, tool_calls -> messages addressed
to functions.*.

One deliberate deviation: vLLM drops analysis messages that precede the latest final-channel assistant message when
it serves the *next* request. Scoring every passage in one forward pass needs them all present, so nothing is
dropped; passages after a final-channel message therefore see slightly more context than at generation time.
The number of rollouts affected is reported.

Usage (athena02): python t2_acts_gptoss.py --model <snapshot> --dom dom/gptoss/dom_vectors.npz --rollouts R.jsonl --out t2/gptoss
"""
import argparse, json, os, time
import numpy as np
import torch
from openai_harmony import Conversation, RenderConversationConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.entrypoints.openai.parser.harmony_utils import (build_harmony_preamble, extract_instructions_from_messages,
                                                         get_encoding, parse_chat_input_to_harmony_message)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dom", required=True)
ap.add_argument("--keys", default="span__all,turn__all,last__all")
ap.add_argument("--rollouts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--attn", default="kernels-community/vllm-flash-attn3")
ap.add_argument("--check-only", action="store_true", help="render + map passages without loading the model")
ap.add_argument("--max-tokens", type=int, default=140000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
si, sn = map(int, a.shard.split("/"))
enc = get_encoding()
MSG_TOK = enc.encode("<|message|>", allowed_special="all")[0]


def to_request(r):
    out = []
    for m in r["chat"]:
        if m["role"] == "assistant":
            x = {"role": "assistant", "content": m["content"] or ""}
            if m.get("reasoning"):
                x["reasoning"] = m["reasoning"]
            if m.get("tool_calls"):
                x["tool_calls"] = [{"id": c["id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                   for c in m["tool_calls"]]
            out.append(x)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id"), "content": m["content"] or ""})
        else:
            out.append({"role": m["role"], "content": m["content"] or ""})
    return out


def render_groups(r):
    """Reproduce what vLLM served: for step j, analysis messages at or before the latest final-channel step F < j
    are dropped (auto_drop_analysis_messages). Steps sharing the same F share one context, so each group is one
    forward pass. Yields (token ids, {(step, channel): (tok0, tok1)}) per group."""
    msgs = to_request(r)
    instructions, rest = extract_instructions_from_messages(msgs)
    tools = [ChatCompletionToolsParam(**t) for t in (r.get("tools") or [])] or None
    pre = build_harmony_preamble(instructions=instructions, tools=tools, reasoning_effort=None,
                                 with_custom_tools=tools is not None)
    tool_id_names = {c["id"]: c["function"]["name"] for m in rest if m["role"] == "assistant" for c in m.get("tool_calls", [])}
    tagged, step = [], 0  # (step, is_assistant, harmony message)
    for m in rest:
        if m["role"] == "assistant":
            step += 1
        for h in parse_chat_input_to_harmony_message(m, tool_id_names):
            tagged.append((step, m["role"] == "assistant", h))
    finals = sorted({s for s, asst, h in tagged if asst and h.channel == "final"})
    groups = {}
    for j in range(1, step + 1):
        groups.setdefault(max([f for f in finals if f < j], default=0), []).append(j)
    n_sys = len(enc.render(pre[0]))
    for F, steps in groups.items():
        last = max(steps)
        sel = [(s, asst, h) for s, asst, h in tagged
               if (s < last or (s == last and asst)) and not (asst and h.channel == "analysis" and s <= F)]
        ids, regions = [], {}
        for h in pre:
            ids += enc.render(h)
        for s, asst, h in sel:
            t = enc.render(h)
            if asst and s in steps:
                k = t.index(MSG_TOK) + 1 if MSG_TOK in t else 0
                body, whole = (len(ids) + k, len(ids) + len(t) - 1), (len(ids), len(ids) + len(t))
                if h.channel == "analysis":
                    regions[(s, "reasoning")] = body
                elif h.recipient and h.recipient.startswith("functions."):
                    g = regions.get((s, "tool_calls"))
                    regions[(s, "tool_calls")] = (g[0] if g else whole[0], whole[1])
                else:
                    regions[(s, "content")] = body
            ids += t
        # The conversation-level render (what vLLM feeds the model) adds a tool-channel note to the system message;
        # everything after it is identical, so shift per-message offsets by the length difference.
        ref = list(enc.render_conversation(Conversation.from_messages(pre + [h for _, _, h in sel]),
                                           config=RenderConversationConfig(auto_drop_analysis=False)))
        delta = len(ref) - len(ids)
        assert ref[n_sys + delta:] == ids[n_sys:], "rendering differs beyond the system message"
        yield ref, {k: (x + delta, y + delta) for k, (x, y) in regions.items()}


rows = [json.loads(l) for i, l in enumerate(open(a.rollouts)) if i % sn == si]
rows = [r for r in rows if r["passages"]]

if a.check_only:
    bad = tot = aff = 0
    for r in rows:
        G = list(render_groups(r)); aff += len(G) > 1
        look = {k: (ids, g) for ids, reg in G for k, g in reg.items()}
        for p in r["passages"]:
            tot += 1
            ids, g = look.get((p["step"], p["channel"]), (None, None))
            if g is None:
                bad += 1; print("missing", r["uid"], p["n"], p["channel"]); continue
            if p["channel"] != "tool_calls":
                got = enc.decode(ids[g[0]:g[1]]).strip()
                if " ".join(got.split()) != " ".join(p["text"].split()):
                    bad += 1
                    if bad < 4: print("MISMATCH", r["uid"], p["n"], p["channel"], "\n got:", got[:150], "\n want:", p["text"][:150])
    print(f"{bad}/{tot} passages unmatched; {aff}/{len(rows)} rollouts need >1 forward pass")
    raise SystemExit

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0",
                                             attn_implementation=a.attn).eval()
keys = a.keys.split(",")
D = np.load(a.dom)
V = torch.tensor(np.stack([D[k] for k in keys], -1), dtype=torch.float32, device="cuda:0")
proj = [None] * V.shape[0]


def mk_hook(L):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        proj[L] = (h[0].float() @ V[L]).cpu()
    return hook


for L, layer in enumerate(model.model.layers):
    layer.register_forward_hook(mk_hook(L))

MX, MN, index, skipped, n_aff = [], [], [], [], 0
t0 = time.time()
with torch.no_grad():
    for j, r in enumerate(rows):
        G = list(render_groups(r)); n_aff += len(G) > 1
        if max(len(ids) for ids, _ in G) > a.max_tokens:
            skipped.append({"uid": r["uid"], "why": "too long"}); continue
        found = {}
        for ids, reg in G:
            model(torch.tensor([ids], device="cuda:0"), use_cache=False, logits_to_keep=1)
            P = torch.stack(proj, 1).numpy()
            for k, g in reg.items():
                if g[1] > g[0]:
                    t = np.arange(g[0], g[1]); found[k] = (P[t].max(0), P[t].mean(0), t, P[t].argmax(0))
        for p in r["passages"]:
            f = found.get((p["step"], p["channel"]))
            if f is None:
                skipped.append({"uid": r["uid"], "n": p["n"], "why": "region not found"}); continue
            MX.append(f[0]); MN.append(f[1]); t = f[2]
            index.append({"uid": r["uid"], "n": p["n"], "step": p["step"], "channel": p["channel"],
                          "tok0": int(t[0]), "tok1": int(t[-1]) + 1, "argmax": (t[f[3]]).tolist(), "n_groups": len(G)})
        if j % 10 == 0:
            print(f"  {j + 1}/{len(rows)} rollouts, {len(ids)} tokens, {time.time() - t0:.0f}s", flush=True)

np.savez(f"{a.out}/scores_{si}.npz", max=np.stack(MX).astype(np.float32), mean=np.stack(MN).astype(np.float32),
         keys=np.array(keys))
with open(f"{a.out}/index_{si}.jsonl", "w") as f:
    for x in index:
        f.write(json.dumps(x) + "\n")
json.dump(skipped, open(f"{a.out}/skipped_{si}.json", "w"), indent=1)
print(f"DONE {len(index)} passages, {len(skipped)} skipped, {n_aff} rollouts needing >1 pass, {time.time() - t0:.0f}s")
