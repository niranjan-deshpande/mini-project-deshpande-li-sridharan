#!/usr/bin/env python3
"""Put the OLMo reward-hacking DoM direction through the J-lens (native finite-difference) and the borrowed
OLMo J++ lens, next to the logit lens. Runs ON the GPU box. Uses Niranjan's lenses.py (fetched from his branch).

Comparison design (all three lenses decode the SAME vector at the SAME transport layer):
  * transport block L -> the vector living at that block's OUTPUT is hs[L+1] in Juliana's extractor
    (hs[0]=emb), i.e. her "L{L+1}". So fd_lens(layer=L) consumes the vector named span_L{L+1}/turn_L{L+1}.
  * J++ (koayon/jpp-lenses, olmo-3-1125-32b) is fitted at blocks 8/16/24/32/40/48/56; we use {32,40,48},
    which bracket our logit-lens cheating band. Those need DoM re-harvested at hs {33,41,49} (new).
  * block 27 (= her L28, the cleanest logit-lens cheating layer) is decoded with logit + native J only
    (not a J++-fitted block); its vectors come from the saved dom npz, no re-harvest.

Self-checks (abort before the expensive loop if any fails):
  A  suffix plumbing: unembed(run_suffix(run_prefix(ids,L),L))[-1] ~= model(ids).logits[-1]  (validates
     _skip_blocks/_PassThrough on OLMo + transformers 5.x; top-1 token must match, cosine > 0.999).
  B  unembed: logit_lens(saved span_L28) overlaps the saved logit_lens.json for span_L28 (>=10/15).
  C  harvest: re-harvested span_L40 cosine vs saved dom span_L40 > 0.99 (render/token-span faithful).
"""
import argparse, json, time, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import lenses
from lenses import (logit_lens, fd_lens, matrix_lens, load_jpp_layer, wikitext_texts,
                    lens_agreement, unembed, format_readouts)

# transport block -> the DoM vector names (Juliana's L = block+1) decoded at that block.
# Block 27 (= her L28) is the cleanest logit-lens cheating layer (bypass/fake/cheat/dummy/pretend) -> native
# J-lens there directly tests whether the J-lens confirms those words at the exact layer we reported them;
# J++ is not fitted at 27, so it gets logit + native-J only (the `if L in JPP_LAYERS` guard handles that).
# Blocks {32,40,48} are the J++-fitted layers bracketing the band -> clean 3-way comparison at identical layers.
FD_LAYERS = {27: ["span_L28", "turn_L28"],   # headline logit-lens layer; from saved dom; logit + native-J
             32: ["span_L33", "turn_L33"],   # J++-fitted
             40: ["span_L41", "turn_L41"],   # J++-fitted (+ random baselines here)
             48: ["span_L49", "turn_L49"]}   # J++-fitted
JPP_LAYERS = [32, 40, 48]
HARVEST_LAYERS = [33, 40, 41, 49]            # 40 is only for self-check C
# logit-lens-only continuity sweep over the band we already report (saved dom, no suffix passes = cheap)
LOGIT_BAND = ["span_L28", "span_L32", "span_L36", "span_L40", "span_L44", "span_L48",
              "turn_L28", "turn_L32", "turn_L36", "turn_L40", "turn_L44", "turn_L48"]
N_BASELINES = 3
JPP_FILE = "olmo-3-1125-32b/lens.pt"


# -------- re-harvest DoM at the hs-indices J++ needs (render/token-span copied from extract_and_score.py) ----
def render(tok, messages):
    norm = []
    for m in messages:
        role = m["role"]
        content = m.get("content", m.get("text", ""))
        if role == "tool":
            role, content = "user", f"[tool output]\n{content}"
        norm.append({"role": role, "content": content, "_orig": m["role"]})
    text = tok.apply_chat_template(norm, tokenize=False, add_generation_prompt=False)
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
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    off = enc["offset_mapping"]
    c0 = text.find(substring)
    if c0 < 0:
        return enc["input_ids"], None
    c1 = c0 + len(substring)
    idxs = [i for i, (a, b) in enumerate(off) if b > c0 and a < c1]
    return enc["input_ids"], idxs


@torch.no_grad()
def reharvest_dom(model, tok, pairs_path, layers, max_len=16384):
    """Returns {f'{harvest}_L{L}': vec} for harvest in span/turn, L in `layers` (hs index)."""
    synth = [json.loads(l) for l in open(pairs_path) if l.strip() and "error" not in l]
    rows = {h: {L: {"hack": [], "honest": []} for L in layers} for h in ("span", "turn")}
    for j, item in enumerate(synth):
        for branch in ("honest", "hack"):
            text, _ = render(tok, item[branch]["messages"])
            span = item[branch]["span"]
            ids, span_idx = token_span_indices(tok, text, span)
            ids = ids[:max_len]
            span_idx = [i for i in (span_idx or []) if i < len(ids)]
            turn_text = next((m["content"] for m in item[branch]["messages"]
                              if m["role"] == "assistant" and span in m["content"]), None)
            _, turn_idx = token_span_indices(tok, text, turn_text) if turn_text else (None, None)
            turn_idx = [i for i in (turn_idx or []) if i < len(ids)]
            if not span_idx and not turn_idx:
                continue
            out = model(torch.tensor([ids], device=model.device), output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            for L in layers:
                if span_idx:
                    rows["span"][L][branch].append(hs[L][0, span_idx, :].float().mean(0).cpu().numpy())
                if turn_idx:
                    rows["turn"][L][branch].append(hs[L][0, turn_idx, :].float().mean(0).cpu().numpy())
        if j % 40 == 0:
            print(f"  reharvest {j}/{len(synth)}", flush=True)
    dom = {}
    for harvest in ("span", "turn"):
        for L in layers:
            H = np.stack(rows[harvest][L]["hack"]); N = np.stack(rows[harvest][L]["honest"])
            dom[f"{harvest}_L{L}"] = (H.mean(0) - N.mean(0)).astype(np.float32)
    return dom


def strip_vecs(res):
    """Drop the fp32 'vec' tensors; keep top/bottom token rows for JSON."""
    return {n: {"top": r["top"], "bottom": r["bottom"]} for n, r in res.items() if not n.startswith("_")}


def cos(a, b):
    a = torch.as_tensor(a).double().reshape(-1); b = torch.as_tensor(b).double().reshape(-1)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dom", required=True, help="saved dom_vectors.npz for this model")
    ap.add_argument("--existing-logit", required=True, help="saved logit_lens.json for self-check B")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--jpp", default="koayon/jpp-lenses")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-wiki", type=int, default=32)
    ap.add_argument("--n-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dir-batch", type=int, default=2)
    args = ap.parse_args()
    t_start = time.time()
    report = {"tag": args.tag, "model": args.model, "config": vars(args), "self_checks": {}, "layers": {},
              "timings": {}}

    print(f"[{args.tag}] loading model ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    model.eval()
    p = lenses._parts(model)
    print(f"  parts: n_layers={p.n_layers} d_model={p.d_model}", flush=True)
    assert p.d_model == 5120, f"expected d_model 5120, got {p.d_model}"

    saved = dict(np.load(args.dom, allow_pickle=True))
    existing_logit = json.load(open(args.existing_logit))
    wiki = wikitext_texts(n=args.n_wiki, n_tokens=args.n_tokens, tok=tok)
    print(f"  wikitext texts: {len(wiki)}", flush=True)

    # ---------------- self-check A: suffix plumbing on OLMo ----------------
    ids = torch.tensor([tok(wiki[0], add_special_tokens=True, truncation=True,
                            max_length=args.n_tokens).input_ids], device=p.text_model.embed_tokens.weight.device)
    with torch.no_grad():
        true_lg = model(ids, use_cache=False).logits[0, -1].float().cpu()
        h = lenses._run_prefix(p, ids, 40)
        hfin = lenses._run_suffix(p, h, 40)
        recon_lg = unembed(model, hfin[:, -1])[0].cpu()
    a_cos = cos(recon_lg, true_lg)
    a_top1 = int(recon_lg.argmax()) == int(true_lg.argmax())
    report["self_checks"]["A_suffix_cos"] = a_cos
    report["self_checks"]["A_top1_match"] = a_top1
    print(f"  [checkA] suffix-recon cos={a_cos:.5f} top1_match={a_top1}", flush=True)
    assert a_cos > 0.999 and a_top1, f"self-check A FAILED (cos {a_cos}, top1 {a_top1}) -> _PassThrough/suffix broken on this model"

    # ---------------- self-check B: unembed matches the saved logit lens ----------------
    llB = logit_lens(model, tok, {"span_L28": saved["span_L28"]}, k=15, filter_tokens=False)
    got = [t[0] for t in llB["span_L28"]["top"]]
    want = existing_logit.get("span_L28", [])
    overlap = len(set(got) & set(want))
    report["self_checks"]["B_logit_overlap"] = f"{overlap}/15"
    report["self_checks"]["B_got"] = got
    print(f"  [checkB] logit-lens span_L28 overlap with saved = {overlap}/15\n    got={got}\n    want={want}", flush=True)
    assert overlap >= 10, f"self-check B FAILED (overlap {overlap}/15) -> unembed/_parts wrong for this model"

    # ---------------- re-harvest DoM at hs {33,40,41,49} ----------------
    print(f"[{args.tag}] re-harvesting DoM at hs {HARVEST_LAYERS} ...", flush=True)
    t0 = time.time()
    new_dom = reharvest_dom(model, tok, args.pairs, HARVEST_LAYERS)
    report["timings"]["reharvest_s"] = time.time() - t0

    # ---------------- self-check C: harvest reproduces saved span_L40 ----------------
    c_cos = cos(new_dom["span_L40"], saved["span_L40"])
    report["self_checks"]["C_harvest_cos_span_L40"] = c_cos
    print(f"  [checkC] re-harvest span_L40 cos vs saved = {c_cos:.4f}", flush=True)
    assert c_cos > 0.99, f"self-check C FAILED (cos {c_cos}) -> re-harvest not faithful"

    # ---------------- assemble vectors ----------------
    vecs = {}
    for names in FD_LAYERS.values():
        for n in names:
            vecs[n] = saved[n] if n in saved else new_dom[n]
    rng = np.random.default_rng(0)
    for i in range(N_BASELINES):
        v = rng.standard_normal(p.d_model).astype(np.float32)
        vecs[f"rand{i}"] = v / np.linalg.norm(v)

    # ---------------- lenses, per transport block ----------------
    for L, names in FD_LAYERS.items():
        use = list(names) + ([f"rand{i}" for i in range(N_BASELINES)] if L == 40 else [])
        Vsub = {n: vecs[n] for n in use}
        entry = {}
        # logit lens (filtered) on these vectors
        ll = logit_lens(model, tok, Vsub, k=20, filter_tokens=True)
        # native J-lens (finite difference) at this block
        print(f"[{args.tag}] FD J-lens at block {L} on {use} ...", flush=True)
        t0 = time.time()
        jl, jstats = fd_lens(model, tok, Vsub, layer=L, mode="j", texts=wiki, n_tokens=args.n_tokens,
                             batch_size=args.batch_size, dir_batch=args.dir_batch, k=20,
                             filter_tokens=True, return_stats=True)
        report["timings"][f"fd_block{L}_s"] = time.time() - t0
        print(f"    {jstats['seconds']:.0f}s ({jstats['seconds_per_direction']:.0f}s/dir, eps={jstats['eps']:.3f})", flush=True)
        entry["logit"] = strip_vecs(ll)
        entry["jlens_fd"] = strip_vecs(jl)
        entry["agreement"] = {"logit_vs_j": lens_agreement(ll, jl, model=model).get("_summary", {})}
        # borrowed J++ at the fitted blocks
        if L in JPP_LAYERS:
            J = load_jpp_layer(args.jpp, L, filename=JPP_FILE)
            jpp = matrix_lens(J, model, tok, Vsub, k=20, filter_tokens=True)
            entry["jpp"] = strip_vecs(jpp)
            entry["agreement"]["j_vs_jpp"] = lens_agreement(jl, jpp, model=model).get("_summary", {})
            entry["agreement"]["logit_vs_jpp"] = lens_agreement(ll, jpp, model=model).get("_summary", {})
            del J
        report["layers"][str(L)] = entry
        # console peek
        print(f"  --- block {L} readouts ---", flush=True)
        print("  [logit]\n" + format_readouts(ll, n=8), flush=True)
        print("  [J-lens]\n" + format_readouts(jl, n=8), flush=True)
        if L in JPP_LAYERS:
            print("  [J++]\n" + format_readouts(jpp, n=8), flush=True)

    # ---------------- logit-lens continuity band (filtered), from saved dom, no suffix passes ----------------
    band = {n: saved[n] for n in LOGIT_BAND if n in saved}
    report["logit_band"] = strip_vecs(logit_lens(model, tok, band, k=15, filter_tokens=True))

    report["timings"]["total_s"] = time.time() - t_start
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"[{args.tag}] DONE -> {args.out}  ({report['timings']['total_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
