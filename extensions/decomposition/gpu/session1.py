"""GPU session 1: pipeline check, direction activations, rollout pass, lenses. One unattended, resumable job.

Stages (each writes to --out and is skipped if its done-marker exists):
  check    rebuild the Claude v_hack from pairs_full.jsonl and compare with Agastya's vector; rescore a few
           rollouts with Agastya's vector and compare with his saved passage scores. Aborts on mismatch.
  pairs    per-pair activations (turn mean, span mean; plus last token for gc_judgment) at --layers for every
           direction's final pair set.
  dirs     DoM vectors per direction at --layer (turn primary, span secondary) + balanced-shuffle nulls for v_hack.
  rollouts layer-L activations for every assistant token (fp16 cache) and per-token projections onto every
           direction, for all rollouts, plus each passage's token range.
  lenses   logit lens, native J-/R-lens readouts (gpu/lenses.py) for every direction.
  manifest sha256 of every output file.
Usage: python gpu/session1.py --model /workspace/models/Qwen3.8-27B --data /workspace/data --out /workspace/out
"""
import argparse, gzip, hashlib, json, os, re, sys, time, traceback
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from rhdecomp.gen.tokcheck import to_chat as pair_to_chat, locate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True, help="dir with final/, pairs_full.jsonl, qwen_rollouts.jsonl.gz, agastya/")
ap.add_argument("--out", required=True)
ap.add_argument("--layer", type=int, default=40)
ap.add_argument("--layers", default="24,32,40", help="layers saved for pair activations")
ap.add_argument("--stages", default="check,pairs,dirs,rollouts,lenses,manifest")
ap.add_argument("--tok-budget", type=int, default=32768, help="tokens per padded batch for pair harvesting")
ap.add_argument("--limit", type=int, default=0, help="debug: cap pairs per direction and rollouts")
ap.add_argument("--test", action="store_true", help="small-model plumbing test: don't enforce check thresholds")
ap.add_argument("--device", default="cuda")
ap.add_argument("--dtype", default="bfloat16")
a = ap.parse_args()
OUT, DATA = Path(a.out), Path(a.data)
OUT.mkdir(parents=True, exist_ok=True)
L = a.layer
LAYERS = sorted({int(x) for x in a.layers.split(",")} | {L})
LOGF = open(OUT / "session1.log", "a", buffering=1)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); LOGF.write(line + "\n")


# ---------------------------------------------------------------- model + hooks
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map=a.device).eval()
lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
N_LAYERS = len(lm.layers)
DEV = a.device
log(f"loaded {a.model}: {N_LAYERS} layers, hidden {model.config.get_text_config().hidden_size}, "
    f"mem {torch.cuda.memory_allocated() / 1e9 if DEV == 'cuda' else 0:.1f} GB")
if DEV == "cuda":
    # fail loudly instead of silently using the slow, memory-hungry torch fallback for the linear-attention layers
    import importlib.util
    assert importlib.util.find_spec("fla") is not None, "flash-linear-attention (fla) is not installed"


class StopForward(Exception):
    pass


class Capture:
    """Forward hooks that keep the outputs of chosen layers and optionally stop the forward pass after the last."""
    def __init__(self, layers, stop=True):
        self.layers, self.stop, self.out = set(layers), stop, {}
        self.last = max(layers)
        self.handles = [lm.layers[i].register_forward_hook(self._hook(i)) for i in self.layers]

    def _hook(self, i):
        def f(mod, inp, out):
            self.out[i] = out[0] if isinstance(out, tuple) else out
            if self.stop and i == self.last:
                raise StopForward
        return f

    def run(self, **kw):
        self.out = {}
        try:
            model(**kw, use_cache=False, logits_to_keep=1)
        except StopForward:
            pass
        return self.out

    def close(self):
        for h in self.handles:
            h.remove()


def done(stage):
    return (OUT / f".{stage}.done").exists()


def mark(stage):
    (OUT / f".{stage}.done").write_text(time.strftime("%c"))


# ---------------------------------------------------------------- pair harvesting
def gc_items(p):
    """General-cheating pairs -> per-branch (ids, positions) using the Qwen chat template."""
    g, fmt = p["gc"], p["family"]
    out = {}
    for side, text, span in (("hack", g["text_a"], g["span_a"]), ("honest", g["text_b"], g["span_b"])):
        if fmt == "gc_ai_offdomain":
            msgs = [{"role": "user", "content": g["shared_context"]}, {"role": "assistant", "content": text}]
            r = locate(tok, {"messages": msgs, "span": span, "turn_index": 1})
            if r is None:
                return None
            out[side] = r; continue
        content = text + ("\n\nWas this fair?" if fmt == "gc_judgment" else "")
        rendered = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                           add_generation_prompt=(fmt == "gc_judgment"))
        t0 = rendered.find(text); t1 = t0 + len(text)
        s0 = rendered.find(span, t0); s1 = s0 + len(span)
        enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
        off = enc["offset_mapping"]
        turn = [i for i, (x, y) in enumerate(off) if y > t0 and x < t1]
        sp = [i for i, (x, y) in enumerate(off) if y > s0 and x < s1]
        if t0 < 0 or s0 < 0 or not turn or not sp:
            return None
        pos = {"turn": turn, "span": sp, "last": [len(enc["input_ids"]) - 1]}
        out[side] = (enc["input_ids"], pos)
    return out


def claude_items(p):
    """pairs_full.jsonl pairs -> same filter and positions as build_dom.py (divergent assistant turn containing span)."""
    H, N = p["hack"]["messages"], p["honest"]["messages"]
    k = next((i for i, (h, n) in enumerate(zip(H, N)) if h != n), None)
    if k is None or H[k]["role"] != "assistant":
        return None
    if not all(p[b]["span"] in p[b]["messages"][k]["content"] for b in ("hack", "honest")):
        return None
    out = {}
    for b in ("hack", "honest"):
        r = locate(tok, {"messages": p[b]["messages"], "span": p[b]["span"], "turn_index": k})
        if r is None:
            return None
        out[b] = r
    return out


def harvest(items, layers, keys=("turn", "span", "last")):
    """items: list of (ids, positions). Returns {key: [n, len(layers), d] float16}. Right-padded batches by length."""
    order = np.argsort([len(x[0]) for x in items])[::-1]
    res = {k: [None] * len(items) for k in keys}
    cap = Capture(layers)
    i = 0
    with torch.no_grad():
        while i < len(order):
            mx = len(items[order[i]][0])
            bs = max(1, a.tok_budget // mx)
            idx = order[i:i + bs]
            ids = torch.full((len(idx), mx), tok.pad_token_id or 0, dtype=torch.long)
            att = torch.zeros((len(idx), mx), dtype=torch.long)
            for r, j in enumerate(idx):
                n = len(items[j][0]); ids[r, :n] = torch.tensor(items[j][0]); att[r, :n] = 1
            outs = cap.run(input_ids=ids.to(DEV), attention_mask=att.to(DEV))
            H = torch.stack([outs[l] for l in layers], 1)  # [b, nl, T, d]
            for r, j in enumerate(idx):
                for k in keys:
                    pos = items[j][1].get(k)
                    if pos:
                        res[k][j] = H[r, :, pos].float().mean(1).cpu().numpy().astype(np.float16)
            i += bs
    cap.close()
    return {k: np.stack(v) for k, v in res.items() if all(x is not None for x in v)}


def harvest_pairs(pairs, fn):
    kept, items = [], []
    for p in pairs:
        it = fn(p)
        if it is None:
            continue
        kept.append(p); items += [it["hack"], it["honest"]]
    acts = harvest(items, LAYERS)
    return kept, {f"{k}_{b}": v[i::2] for k, v in acts.items() for i, b in enumerate(("hack", "honest"))}


def cos(x, y):
    return float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))


# ---------------------------------------------------------------- rollouts (render exactly as agastya t2_acts.py)
def roll_chat(r):
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
    out, pos, asst = {}, 0, []
    for step in range(1, n_steps + 1):
        t0 = text.find("<|im_start|>assistant", pos)
        if t0 < 0:
            break
        a0 = t0
        t0 += len("<|im_start|>assistant")
        t1 = text.find("<|im_end|>", t0); t1 = len(text) if t1 < 0 else t1
        pos = t1; asst.append((a0, t1))
        o, c = text.find("<think>", t0, t1), text.find("</think>", t0, t1)
        body = t0
        if o >= 0 and c >= 0:
            out[(step, "reasoning")] = (o + len("<think>"), c); body = c + len("</think>")
        tc = text.find("<tool_call>", body, t1)
        out[(step, "content")] = (body, tc if tc >= 0 else t1)
        if tc >= 0:
            out[(step, "tool_calls")] = (tc, t1)
    return out, asst


def score_rollout(r, V):
    """V: [d, K] tensor. Returns (asst positions, acts fp16 [T_a, d], proj fp32 [T_a, K], passages)."""
    text = tok.apply_chat_template(roll_chat(r), tools=r.get("tools") or None, tokenize=False,
                                   add_generation_prompt=False, preserve_thinking=True)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, off = enc["input_ids"], np.array(enc["offset_mapping"])
    reg, asst = regions(text, r["n_steps"])
    amask = np.zeros(len(ids), bool)
    for s0, s1 in asst:
        amask |= (off[:, 1] > s0) & (off[:, 0] < s1)
    cap = Capture([L])
    with torch.no_grad():
        h = cap.run(input_ids=torch.tensor([ids], device=DEV))[L][0]  # [T, d]
    cap.close()
    apos = np.flatnonzero(amask)
    ha = h[torch.tensor(apos, device=DEV)]
    proj = (ha.float() @ V).cpu().numpy()
    passages = []
    for p in r["passages"]:
        g = reg.get((p["step"], p["channel"]))
        if g is None:
            continue
        s0, s1 = g
        while s0 < s1 and text[s0].isspace(): s0 += 1
        while s1 > s0 and text[s1 - 1].isspace(): s1 -= 1
        t = np.where((off[:, 1] > s0) & (off[:, 0] < s1))[0]
        if len(t):
            passages.append({"n": p["n"], "step": p["step"], "channel": p["channel"], "tok0": int(t[0]), "tok1": int(t[-1]) + 1})
    return apos, ha.to(torch.float16).cpu().numpy(), proj, passages, len(ids)


# ---------------------------------------------------------------- stages
def stage_check():
    D = np.load(DATA / "agastya/dom_vectors.npz")
    v_ag = D["turn__all"][L].astype(np.float32)
    pairs = [json.loads(l) for l in open(DATA / "pairs_full.jsonl") if l.strip()]
    if a.limit:
        pairs = pairs[: a.limit]
    kept, acts = harvest_pairs(pairs, claude_items)
    li = LAYERS.index(L)
    v_cl = (acts["turn_hack"][:, li].astype(np.float32) - acts["turn_honest"][:, li].astype(np.float32)).mean(0)
    np.savez(OUT / "claude_pair_acts.npz", **acts, families=np.array([p["family"] for p in kept]), layers=np.array(LAYERS))
    c = cos(v_cl, v_ag)
    log(f"check: rebuilt Claude v_hack from {len(kept)} pairs; cos with Agastya's turn__all L{L} = {c:.5f}")
    # rescore rollouts with Agastya's vector and compare passage max scores
    ag_scores = {}
    for part in sorted((DATA / "agastya/t2").glob("qwen_part*")):
        z = np.load(part / "scores_0.npz"); keys = list(z["keys"]); mx = z["max"][:, L, keys.index("turn__all")]
        for i, l in enumerate(open(part / "index_0.jsonl")):
            x = json.loads(l); ag_scores[(x["uid"], x["n"])] = float(mx[i])
    rolls = [json.loads(l) for l in gzip.open(DATA / "qwen_rollouts.jsonl.gz", "rt")]
    rolls = sorted([r for r in rolls if r["passages"]], key=lambda r: len(r["judge_text"]))[:: max(1, len(rolls) // 6)][:6]
    V = torch.tensor(v_ag[:, None], device=DEV)
    ours, theirs = [], []
    for r in rolls:
        apos, _, proj, passages, T = score_rollout(r, V)
        where = {p: i for i, p in enumerate(apos)}
        for p in passages:
            idx = [where[t] for t in range(p["tok0"], p["tok1"]) if t in where]
            if idx and (r["uid"], p["n"]) in ag_scores:
                ours.append(proj[idx, 0].max()); theirs.append(ag_scores[(r["uid"], p["n"])])
    ours, theirs = np.array(ours), np.array(theirs)
    corr = float(np.corrcoef(ours, theirs)[0, 1]); rel = float(np.median(np.abs(ours - theirs) / (np.abs(theirs) + 1e-6)))
    res = {"cos_claude_rebuilt_vs_agastya": c, "n_claude_pairs": len(kept), "rescore_passages": len(ours),
           "rescore_corr": corr, "rescore_median_rel_diff": rel}
    json.dump(res, open(OUT / "check.json", "w"), indent=1)
    log(f"check: rescored {len(ours)} passages from {len(rolls)} rollouts: corr {corr:.5f}, median rel diff {rel:.4f}")
    if not a.test and (c < 0.99 or corr < 0.99):
        raise RuntimeError(f"pipeline check failed: {res}")


def stage_pairs():
    (OUT / "pairs").mkdir(exist_ok=True)
    for f in sorted((DATA / "final").glob("pairs_*.jsonl")):
        d = f.stem[len("pairs_"):]
        dst = OUT / "pairs" / f"{d}.npz"
        if dst.exists():
            continue
        pairs = [json.loads(l) for l in open(f)]
        if a.limit:
            pairs = pairs[: a.limit]
        fn = gc_items if d.startswith("gc_") else (lambda p: {b: locate(tok, p[b]) for b in ("hack", "honest")}
                                                    if all(locate(tok, p[b]) for b in ("hack", "honest")) else None)
        t0 = time.time()
        kept, acts = harvest_pairs(pairs, fn)
        np.savez(dst, **acts, ids=np.array([f"{d}_{p['meta']['i']:03d}" for p in kept]), layers=np.array(LAYERS))
        log(f"pairs: {d}: {len(kept)}/{len(pairs)} pairs harvested in {time.time() - t0:.0f}s")


def stage_dirs():
    li = LAYERS.index(L)
    vecs, names = [], []
    for f in sorted((OUT / "pairs").glob("*.npz")):
        z = np.load(f); d = f.stem
        for k in ("turn", "span"):
            A, B = z[f"{k}_hack"][:, li].astype(np.float32), z[f"{k}_honest"][:, li].astype(np.float32)
            vecs.append(A.mean(0) - B.mean(0)); names.append(f"{d}__{k}")
        if d.startswith("gc_judgment"):
            vecs.append(z["last_hack"][:, li].astype(np.float32).mean(0) - z["last_honest"][:, li].astype(np.float32).mean(0))
            names.append(f"{d}__last")
    # balanced sign-flip nulls of v_hack (turn): exactly half the pairs swapped
    z = np.load(OUT / "pairs" / "hack.npz")
    Dh = z["turn_hack"][:, li].astype(np.float32) - z["turn_honest"][:, li].astype(np.float32)
    rng = np.random.default_rng(0)
    for s in range(20):
        eps = np.ones(len(Dh)); eps[rng.permutation(len(Dh))[: len(Dh) // 2]] = -1
        vecs.append((eps[:, None] * Dh).mean(0)); names.append(f"null_hack_shuffle{s:02d}")
    ag = np.load(DATA / "agastya/dom_vectors.npz")
    vecs.append(ag["turn__all"][L].astype(np.float32)); names.append("agastya__turn")
    cz = np.load(OUT / "claude_pair_acts.npz")
    vecs.append((cz["turn_hack"][:, li].astype(np.float32) - cz["turn_honest"][:, li].astype(np.float32)).mean(0))
    names.append("claude_rebuilt__turn")
    np.savez(OUT / "directions.npz", V=np.stack(vecs), names=np.array(names), layer=L)
    log(f"dirs: {len(names)} directions at L{L}")


def stage_rollouts():
    Z = np.load(OUT / "directions.npz")
    V = torch.tensor(Z["V"].T, device=DEV, dtype=torch.float32)  # [d, K]
    (OUT / "rollouts").mkdir(exist_ok=True); (OUT / "cache").mkdir(exist_ok=True)
    rolls = [json.loads(l) for l in gzip.open(DATA / "qwen_rollouts.jsonl.gz", "rt")]
    rolls = sorted([r for r in rolls if r["passages"]], key=lambda r: -len(r["judge_text"]))  # longest first: OOM shows early
    if a.limit:
        rolls = rolls[: a.limit]
    t0 = time.time()
    for j, r in enumerate(rolls):
        key = re.sub(r"[^\w.-]", "_", r["uid"])
        if (OUT / "rollouts" / f"{key}.npz").exists():
            continue
        apos, acts, proj, passages, T = score_rollout(r, V)
        np.save(OUT / "cache" / f"{key}.npy", acts)
        np.savez(OUT / "rollouts" / f"{key}.npz", pos=apos, proj=proj, names=Z["names"], n_tokens=T)
        (OUT / "rollouts" / f"{key}.passages.json").write_text(json.dumps({"uid": r["uid"], "split": r["split"],
                                                                         "sample_id": r["sample_id"], "passages": passages}))
        if j == 0 and DEV == "cuda":
            log(f"rollouts: longest rollout {T} tokens, peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        if j % 10 == 0:
            log(f"rollouts: {j + 1}/{len(rolls)} ({T} tokens), {time.time() - t0:.0f}s")


def stage_lenses():
    import lenses
    Z = np.load(OUT / "directions.npz")
    keep = [i for i, n in enumerate(Z["names"]) if not n.startswith("null_")]
    V = {str(Z["names"][i]): torch.tensor(Z["V"][i]) for i in keep}
    texts = lenses.wikitext_texts(64, 128, tok)
    res = {"logit": lenses.logit_lens(model, tok, V)}
    for mode in ("j", "r"):
        t0 = time.time()
        res[f"{mode}_lens"] = lenses.fd_lens(model, tok, V, L, texts, mode=mode, eps_frac=0.05)  # 5%: bf16-safe
        log(f"lenses: native {mode}-lens for {len(V)} directions in {time.time() - t0:.0f}s")
    jpp = os.environ.get("JPP_LENS")
    if jpp and Path(jpp).exists():
        J = lenses.load_jpp_layer(jpp, L)
        res["jpp_borrowed"] = lenses.matrix_lens(J, model, tok, V)
        res["agreement_jpp_vs_j"] = lenses.lens_agreement(res["jpp_borrowed"], res["j_lens"], model=model)
        res["agreement_jpp_vs_r"] = lenses.lens_agreement(res["jpp_borrowed"], res["r_lens"], model=model)
        log(f"lenses: borrowed J++ agreement medians: vs J {res['agreement_jpp_vs_j'].get('median')}, "
            f"vs R {res['agreement_jpp_vs_r'].get('median')}")
    json.dump(res, open(OUT / "lenses.json", "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


def stage_manifest():
    man = {}
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.name != "manifest.json" and "cache" not in f.parts:
            man[str(f.relative_to(OUT))] = {"bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    cache = sorted((OUT / "cache").glob("*.npy")) if (OUT / "cache").exists() else []
    man["cache_summary"] = {"files": len(cache), "bytes": sum(f.stat().st_size for f in cache)}
    json.dump(man, open(OUT / "manifest.json", "w"), indent=1)
    log(f"manifest: {len(man) - 1} files (+ {len(cache)} cache files, {man['cache_summary']['bytes'] / 1e9:.1f} GB)")


STAGES = {"check": stage_check, "pairs": stage_pairs, "dirs": stage_dirs, "rollouts": stage_rollouts,
          "lenses": stage_lenses, "manifest": stage_manifest}
try:
    for s in a.stages.split(","):
        if done(s) and s != "manifest":
            log(f"{s}: already done"); continue
        log(f"{s}: start"); STAGES[s](); mark(s); log(f"{s}: done")
    log("SESSION1 COMPLETE")
except Exception:
    log("FAILED:\n" + traceback.format_exc())
    (OUT / "FAILED").write_text(traceback.format_exc())
    sys.exit(1)
