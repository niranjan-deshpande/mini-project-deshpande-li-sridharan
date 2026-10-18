"""T1: build difference-of-means reward-hacking vectors from synthetic minimal pairs + logit-lens readout.

Teacher-forces the model over each hack/honest pair, harvests residual-stream activations at three position sets
in the divergent assistant turn (span = the marked phrase, turn = whole turn, last = final span token), and saves
per-pair activations for every layer plus DoM vectors (all families pooled + per family).

Usage (athena02): python build_dom.py --model <snapshot> --family qwen --out /data/agastyas/rh_probe/dom/qwen
"""
import argparse, collections, json, os, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--family", choices=["qwen", "gptoss", "glm"], required=True)
ap.add_argument("--pairs", default="/data/agastyas/rh_probe/pairs_full.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--topk", type=int, default=15)
ap.add_argument("--harmony", action="store_true", help="gpt-oss: render via vLLM harmony path")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

THINK = re.compile(r"<think>(.*?)</think>\s*", re.S)


def to_chat(messages):
    """Synthetic messages -> model chat format. <think> blocks become the model's native reasoning field."""
    out = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "assistant":
            reasoning = "\n".join(t.strip() for t in THINK.findall(content))
            content = THINK.sub("", content).strip()
            msg = {"role": "assistant", "content": content}
            if reasoning:
                msg["reasoning_content" if a.family in ("qwen", "glm") else "thinking"] = reasoning
            out.append(msg)
        elif role == "tool":
            # Qwen's template renders standalone tool responses; gpt-oss's harmony template needs a matching
            # tool call, so tool output becomes a user turn there (same trick as the 2881r repo).
            out.append({"role": "tool", "content": content} if a.family in ("qwen", "glm")
                       else {"role": "user", "content": f"[tool output]\n{content}"})
        else:
            out.append({"role": role, "content": content})
    return out


def divergence_index(p):
    """Index of the first message where the hack and honest branches differ."""
    H, N = p["hack"]["messages"], p["honest"]["messages"]
    for i, (h, n) in enumerate(zip(H, N)):
        if h != n:
            return i
    return None


tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto" if a.family == "glm" else "cuda:0",
                                             **({"experts_implementation": "grouped_mm"} if a.family == "glm" else {}))
DEV = model.get_input_embeddings().weight.device
model.eval()
ASST_OPEN = {"qwen": "<|im_start|>assistant", "gptoss": "<|start|>assistant", "glm": "<|assistant|>"}[a.family]
TURN_END = "<|im_end|>" if a.family == "qwen" else "<|end|>"

pairs = [json.loads(l) for l in open(a.pairs) if l.strip()]
kept = []
for p in pairs:
    k = divergence_index(p)
    if k is None or p["hack"]["messages"][k]["role"] != "assistant":
        continue
    if not all(p[b]["span"] in p[b]["messages"][k]["content"] for b in ("hack", "honest")):
        continue
    kept.append((p, k))
print(f"pairs: {len(pairs)} total, {len(kept)} with a shared prefix diverging at an assistant turn containing the span")


def render_harmony(msgs):
    """gpt-oss: render through vLLM's harmony path (as `vllm serve` builds prompts) with every analysis message
    kept; HF's chat template drops reasoning from earlier turns, which loses spans located there."""
    from openai_harmony import Conversation, RenderConversationConfig
    from vllm.entrypoints.openai.parser.harmony_utils import (build_harmony_preamble, extract_instructions_from_messages,
                                                             get_encoding, parse_chat_input_to_harmony_message)
    msgs = [{**{k: v for k, v in m.items() if k != "thinking"}, **({"reasoning": m["thinking"]} if "thinking" in m else {})}
            for m in msgs]
    instructions, rest = extract_instructions_from_messages(msgs)
    hs = build_harmony_preamble(instructions=instructions, tools=None, reasoning_effort=None)
    for m in rest:
        hs += parse_chat_input_to_harmony_message(m)
    ids = get_encoding().render_conversation(Conversation.from_messages(hs), config=RenderConversationConfig(auto_drop_analysis=False))
    return get_encoding().decode_utf8(ids)


def render(p, branch, k=None):
    if a.family == "glm":
        # GLM's template shows reasoning only after the last user message, so render up to the divergent turn k;
        # attention is causal, so activations on turn k's tokens are identical to a full render that kept it.
        return tok.apply_chat_template(to_chat(p[branch]["messages"][: k + 1]), tokenize=False, add_generation_prompt=False)
    msgs = to_chat(p[branch]["messages"])
    if a.family == "gptoss" and a.harmony:
        return render_harmony(msgs)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                   **({"preserve_thinking": True} if a.family == "qwen" else {}))


def turn_bounds(text, p, branch, k, d):
    if a.family == "glm":
        n_asst = sum(m["role"] == "assistant" for m in p[branch]["messages"][: k + 1])
        starts = [m.start() for m in re.finditer(re.escape(ASST_OPEN), text)]
        if len(starts) < n_asst:
            return None
        t0 = starts[n_asst - 1]
        nxt = [x for x in (text.find(t, t0 + 1) for t in ("<|user|>", "<|observation|>", "<|system|>", "<|assistant|>")) if x >= 0]
        return t0, min(nxt) if nxt else len(text)
    if a.family == "qwen":
        # the divergent message is the n-th assistant message
        n_asst = sum(m["role"] == "assistant" for m in p[branch]["messages"][: k + 1])
        starts = [m.start() for m in re.finditer(re.escape(ASST_OPEN), text)]
        if len(starts) < n_asst:
            return None
        t0 = starts[n_asst - 1]
        t1 = text.find(TURN_END, t0)
        return t0, len(text) if t1 < 0 else t1
    # harmony splits a turn into analysis/final blocks: the turn starts at the block containing the first
    # hack-vs-honest difference and runs until the next non-assistant block
    t0 = text.rfind("<|start|>", 0, d + 1)
    nxt = [x for x in (text.find(t, d) for t in ("<|start|>user", "<|start|>functions", "<|start|>system")) if x >= 0]
    return t0, min(nxt) if nxt else len(text)


def render_and_locate(p, branch, k, text, d):
    tb = turn_bounds(text, p, branch, k, d)
    if tb is None:
        return None
    t0, t1 = tb
    s0 = text.find(p[branch]["span"], t0)
    if s0 < 0 or s0 > t1:
        return None
    s1 = s0 + len(p[branch]["span"])
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    off = enc["offset_mapping"]
    span = [i for i, (x, y) in enumerate(off) if y > s0 and x < s1]
    turn = [i for i, (x, y) in enumerate(off) if y > t0 and x < t1]
    if not span or not turn:
        return None
    return enc["input_ids"], {"span": span, "turn": turn, "last": [span[-1]]}


HARVESTS = ("span", "turn", "last")
acts = {(h, b): [] for h in HARVESTS for b in ("hack", "honest")}
fams, n_tokens = [], []
with torch.no_grad():
    for j, (p, k) in enumerate(kept):
        texts = {b: render(p, b, k) for b in ("hack", "honest")}
        d = next((i for i, (x, y) in enumerate(zip(texts["hack"], texts["honest"])) if x != y), 0)
        loc = {b: render_and_locate(p, b, k, texts[b], d) for b in ("hack", "honest")}
        if any(v is None for v in loc.values()):
            continue
        for b in ("hack", "honest"):
            ids, pos = loc[b]
            hs = model(torch.tensor([ids], device=DEV), output_hidden_states=True, use_cache=False).hidden_states
            H = torch.stack([h.to(DEV) for h in hs[1:]], 0)[:, 0]  # [layers, seq, d], drop embeddings
            for h in HARVESTS:
                acts[(h, b)].append(H[:, pos[h]].float().mean(1).cpu().numpy().astype(np.float16))
            n_tokens.append(len(ids))
        fams.append(p["family"])
        if j % 25 == 0:
            print(f"  {j}/{len(kept)}", flush=True)

fams = np.array(fams)
print(f"harvested {len(fams)} pairs; families {collections.Counter(fams.tolist())}; tokens/conv mean {np.mean(n_tokens):.0f}")
np.savez_compressed(f"{a.out}/pair_acts.npz", families=fams,
                    **{f"{h}_{b}": np.stack(acts[(h, b)]) for (h, b) in acts})

# DoM vectors: [layers, d] per (harvest, group)
dom = {}
for h in HARVESTS:
    Hk, Nk = np.stack(acts[(h, "hack")]).astype(np.float32), np.stack(acts[(h, "honest")]).astype(np.float32)
    dom[f"{h}__all"] = Hk.mean(0) - Nk.mean(0)
    for f in sorted(set(fams)):
        m = fams == f
        dom[f"{h}__{f}"] = Hk[m].mean(0) - Nk[m].mean(0)
np.savez(f"{a.out}/dom_vectors.npz", **dom)

# held-out separability on synthetic data (pair-level 5-fold): does the DoM direction separate unseen pairs?
rng = np.random.default_rng(0)
folds = rng.permutation(len(fams)) % 5
sep = {}
for h in HARVESTS:
    Hk, Nk = np.stack(acts[(h, "hack")]).astype(np.float32), np.stack(acts[(h, "honest")]).astype(np.float32)
    accs = []
    for L in range(Hk.shape[1]):
        correct = 0
        for f in range(5):
            tr, te = folds != f, folds == f
            v = Hk[tr, L].mean(0) - Nk[tr, L].mean(0)
            correct += int(((Hk[te, L] - Nk[te, L]) @ v > 0).sum())  # paired: hack projects higher than its honest twin
        accs.append(correct / len(fams))
    sep[h] = accs

# logit lens
lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
norm, head = lm.norm, model.get_output_embeddings()
corpus = " ".join(m["content"] for p, _ in kept for b in ("hack", "honest") for m in p[b]["messages"]).lower()
lens = {}
with torch.no_grad():
    for key, V in dom.items():
        lens[key] = {}
        for L in range(V.shape[0]):
            v = torch.tensor(V[L], device=norm.weight.device, dtype=torch.bfloat16)
            logits = head(norm(v[None]).to(head.weight.device))[0].float()
            top = [tok.decode([t]) for t in torch.topk(logits, a.topk).indices.tolist()]
            bot = [tok.decode([t]) for t in torch.topk(-logits, a.topk).indices.tolist()]
            lens[key][L] = {"top": top, "bottom": bot,
                            "top_in_synthetic_text": [t.strip().lower() in corpus for t in top if t.strip()]}
json.dump({"lens": lens, "synthetic_paired_acc": sep, "n_pairs": len(fams)}, open(f"{a.out}/logit_lens.json", "w"), indent=1)

print("\nheld-out paired accuracy on synthetic pairs (every 4th layer):")
for h in HARVESTS:
    print(f"  {h:5}", " ".join(f"L{L}:{sep[h][L]:.2f}" for L in range(3, len(sep[h]), 4)))
print("\nlogit lens, pooled vectors (every 4th layer from L16):")
for h in HARVESTS:
    for L in range(15, dom[f"{h}__all"].shape[0], 4):
        e = lens[f"{h}__all"][L]
        print(f"  {h:5} L{L:2}: {e['top'][:10]}")  # L = index of model.layers[L] whose output this is
print("DONE")
