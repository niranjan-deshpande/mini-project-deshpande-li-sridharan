#!/usr/bin/env python3
"""Build the DoM reward-hacking vector and score rollouts. Runs ON the GPU VM (HF transformers).

Three stages in one GPU session:
  1. SYNTH: forward OLMo over each synthetic minimal pair; extract residual-stream mean over the marked
     span, per candidate layer, for the hack and honest branch.
  2. DOM: v_layer = mean(hack span vecs) - mean(honest span vecs), per layer. (numpy, on VM.)
  3. SCORE: forward OLMo over each rollout; for each layer, score = max over assistant-channel tokens of
     <v_layer, h_t>. Save per-rollout per-layer scores.

Outputs (small, pulled back to laptop): synth_vectors.npz, dom_vectors.npz, rollout_scores.csv.
AUROC / layer selection / logit-lens done offline with the labels.

Usage:
  python extract_and_score.py --model ~/models/olmo-3.1-32b-think \
      --synth synth/pairs_full.jsonl --rollouts transcripts_full.jsonl --outdir ~/probe_out
"""
import argparse, csv, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CANDIDATE_LAYERS = [16, 20, 24, 28, 32, 36, 40, 44, 48, 52]  # mid-depth sweep (OLMo has 64)


def render(tok, messages):
    """Render a conversation to text + per-char role map. tool->user(mapped). Returns (text, char_roles)."""
    # Map 'tool' to a user turn so the OLMo chat template accepts it.
    norm = []
    for m in messages:
        role = m["role"]
        content = m.get("content", m.get("text", ""))  # synth uses 'content', rollouts use 'text'
        if role == "tool":
            role, content = "user", f"[tool output]\n{content}"
        norm.append({"role": role, "content": content, "_orig": m["role"]})
    text = tok.apply_chat_template(norm, tokenize=False, add_generation_prompt=False)
    # char->is_assistant map: mark spans of assistant content inside `text`
    is_asst = bytearray(len(text))
    cur = 0
    for m in norm:
        if m["_orig"] == "assistant":
            idx = text.find(m["content"], cur)
            if idx >= 0:
                for i in range(idx, idx + len(m["content"])):
                    is_asst[i] = 1
                cur = idx + len(m["content"])
    return text, is_asst


def token_span_indices(tok, text, substring):
    """Token indices whose char range overlaps the first occurrence of substring."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    off = enc["offset_mapping"]
    c0 = text.find(substring)
    if c0 < 0:
        return enc["input_ids"], None
    c1 = c0 + len(substring)
    idxs = [i for i, (a, b) in enumerate(off) if b > c0 and a < c1]
    return enc["input_ids"], idxs


def assistant_token_mask(tok, text, is_asst):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    off = enc["offset_mapping"]
    mask = [1 if (b <= len(is_asst) and a < len(is_asst) and any(is_asst[a:b])) else 0 for (a, b) in off]
    return enc["input_ids"], mask


@torch.no_grad()
def hidden_states(model, tok, input_ids):
    ids = torch.tensor([input_ids], device=model.device)
    out = model(ids, output_hidden_states=True, use_cache=False)
    # hidden_states: tuple len 65 (emb + 64 layers); index layer L -> hs[L]
    return out.hidden_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--no-score", action="store_true", help="skip rollout scoring (just build DoM + logit-lens)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    model.eval()

    # ---- Stage 1: synth vectors. Dual harvest: "span" (marked phrase) and "turn" (whole diverging turn) ----
    synth = [json.loads(l) for l in open(args.synth) if l.strip() and "error" not in l]
    rows = {h: {L: {"hack": [], "honest": []} for L in CANDIDATE_LAYERS} for h in ("span", "turn")}
    for j, item in enumerate(synth):
        for branch in ("honest", "hack"):
            text, _ = render(tok, item[branch]["messages"])
            span = item[branch]["span"]
            ids, span_idx = token_span_indices(tok, text, span)
            ids = ids[: args.max_len]
            span_idx = [i for i in (span_idx or []) if i < len(ids)]
            # "turn" = the full assistant turn that contains the span
            turn_text = next((m["content"] for m in item[branch]["messages"]
                              if m["role"] == "assistant" and span in m["content"]), None)
            _, turn_idx = token_span_indices(tok, text, turn_text) if turn_text else (None, None)
            turn_idx = [i for i in (turn_idx or []) if i < len(ids)]
            if not span_idx and not turn_idx:
                continue
            hs = hidden_states(model, tok, ids)
            for L in CANDIDATE_LAYERS:
                if span_idx:
                    rows["span"][L][branch].append(hs[L][0, span_idx, :].float().mean(0).cpu().numpy())
                if turn_idx:
                    rows["turn"][L][branch].append(hs[L][0, turn_idx, :].float().mean(0).cpu().numpy())
        if j % 20 == 0:
            print(f"synth {j}/{len(synth)}", flush=True)

    dom = {}
    synth_save = {}
    for harvest in ("span", "turn"):
        for L in CANDIDATE_LAYERS:
            H = np.stack(rows[harvest][L]["hack"]); N = np.stack(rows[harvest][L]["honest"])
            dom[f"{harvest}_L{L}"] = (H.mean(0) - N.mean(0)).astype(np.float32)
            synth_save[f"{harvest}_L{L}_hack"] = H.astype(np.float32)
            synth_save[f"{harvest}_L{L}_honest"] = N.astype(np.float32)
    # Logistic-regression direction per (harvest, layer): stronger alternative to DoM (paper notes this).
    try:
        from sklearn.linear_model import LogisticRegression
        for harvest in ("span", "turn"):
            for L in CANDIDATE_LAYERS:
                H = synth_save[f"{harvest}_L{L}_hack"]; N = synth_save[f"{harvest}_L{L}_honest"]
                X = np.vstack([H, N]); y = np.r_[np.ones(len(H)), np.zeros(len(N))]
                Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
                clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xn, y)
                w = (clf.coef_[0] / (X.std(0) + 1e-6)).astype(np.float32)  # back to raw-activation space
                dom[f"lr_{harvest}_L{L}"] = (w / (np.linalg.norm(w) + 1e-8)).astype(np.float32)
        print("added LR directions", flush=True)
    except Exception as e:
        print(f"[warn] LR skipped: {e}", flush=True)

    np.savez(os.path.join(args.outdir, "dom_vectors.npz"), **dom)
    np.savez(os.path.join(args.outdir, "synth_vectors.npz"), **synth_save)
    print(f"saved {len(dom)} directions (DoM + LR, span+turn x {len(CANDIDATE_LAYERS)} layers)", flush=True)

    # ---- Stage 2b: logit-lens readout of each DoM vector (Table-1 analogue) ----
    lens = {}
    with torch.no_grad():
        norm = model.model.norm
        lm_head = model.lm_head
        for key, vec in dom.items():
            v = torch.tensor(vec, device=model.device, dtype=torch.bfloat16)
            logits = lm_head(norm(v))
            topk = torch.topk(logits.float(), 15).indices.tolist()
            lens[key] = [tok.decode([t]) for t in topk]
    json.dump(lens, open(os.path.join(args.outdir, "logit_lens.json"), "w"), indent=2)
    print("saved logit-lens (span harvest, all layers):", flush=True)
    for L in CANDIDATE_LAYERS:
        print(f"  span_L{L}: {lens[f'span_L{L}']}", flush=True)

    if args.no_score:
        print("DONE (no-score: DoM + logit-lens only)", flush=True)
        return

    # ---- Stage 3: score rollouts (both harvests x layers, max & mean pooling) ----
    rollouts = [json.loads(l) for l in open(args.rollouts) if l.strip()]
    keys = list(dom.keys())  # span_L16 ... turn_L52
    vt = {k: torch.tensor(dom[k], device=model.device, dtype=torch.bfloat16) for k in keys}
    cols = [f"{k}_{p}" for k in keys for p in ("max", "mean")]
    with open(os.path.join(args.outdir, "rollout_scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "split", "prompt", "score", "errored"] + cols)
        for k, r in enumerate(rollouts):
            text, is_asst = render(tok, r["messages"])
            ids, amask = assistant_token_mask(tok, text, is_asst)
            if len(ids) > args.max_len:               # hacks are LATE -> keep the tail
                ids = ids[-args.max_len:]; amask = amask[-args.max_len:]
            pos = [i for i, m in enumerate(amask) if m] or list(range(len(ids)))
            hs = hidden_states(model, tok, ids)
            vals = []
            for key in keys:
                L = int(key.split("_L")[1])
                h = hs[L][0, pos, :].to(torch.bfloat16)
                s = (h @ vt[key]).float().cpu().numpy()
                vals += [float(s.max()), float(s.mean())]
            w.writerow([r.get("task_id"), r.get("split"), r.get("prompt"), r.get("score"), r.get("errored")] + vals)
            if k % 20 == 0:
                print(f"score {k}/{len(rollouts)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
