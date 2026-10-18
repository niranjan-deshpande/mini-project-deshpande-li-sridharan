"""GPU session 2: emotion vectors (Sofroniew et al. 2026 recipe, adapted; notes/emotions_method.md) and their per-token
projections on the cached layer-L rollout activations from session 1.

  stories  mean layer-L activation per story over tokens from the 50th on (plain text, no chat template)
  neutral  token-level centered PCA of layer-L activations on neutral Person/AI dialogues (Qwen chat template;
           first token and special tokens excluded); keep the top PCs reaching 50% of variance
  vectors  v_e = mean_e - mean over emotions; raw and denoised (top neutral PCs projected out)
  project  per-token projections of every cached rollout onto the unit emotion vectors (raw and denoised)
Usage: python gpu/session2_emotions.py --model ... --data /workspace/data --out /workspace/out
"""
import argparse, json, sys, time, traceback
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True, help="dir with emotions/items.jsonl and emotions/neutral.jsonl")
ap.add_argument("--out", required=True, help="session-1 output dir (has cache/ and rollouts/)")
ap.add_argument("--layer", type=int, default=40)
ap.add_argument("--skip", type=int, default=50, help="average story activations from this token on")
ap.add_argument("--tok-budget", type=int, default=32768)
ap.add_argument("--device", default="cuda")
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
OUT, DATA, L, DEV = Path(a.out), Path(a.data), a.layer, a.device
EO = OUT / "emotions"
EO.mkdir(parents=True, exist_ok=True)
LOGF = open(EO / "session2.log", "a", buffering=1)


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); LOGF.write(line + "\n")


tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map=DEV).eval()
lm = model.model.language_model if hasattr(model.model, "language_model") else model.model


class Stop(Exception):
    pass


store = {}


def hook(mod, inp, out):
    store["h"] = out[0] if isinstance(out, tuple) else out
    raise Stop


lm.layers[L].register_forward_hook(hook)


def forward(batch_ids):
    """Right-padded batch -> layer-L outputs [b, T, d]."""
    mx = max(len(x) for x in batch_ids)
    ids = torch.full((len(batch_ids), mx), tok.pad_token_id or 0, dtype=torch.long)
    att = torch.zeros_like(ids)
    for r, x in enumerate(batch_ids):
        ids[r, :len(x)] = torch.tensor(x); att[r, :len(x)] = 1
    try:
        with torch.no_grad():
            model(input_ids=ids.to(DEV), attention_mask=att.to(DEV), use_cache=False, logits_to_keep=1)
    except Stop:
        pass
    return store["h"]


def batches(seqs):
    order = np.argsort([len(s) for s in seqs])[::-1]
    i = 0
    while i < len(order):
        bs = max(1, a.tok_budget // len(seqs[order[i]]))
        yield order[i:i + bs]
        i += bs


def stage_stories():
    items = [json.loads(l) for l in open(DATA / "emotions/items.jsonl")]
    if a.limit:
        items = items[: a.limit]
    seqs = [tok(x["text"], add_special_tokens=False)["input_ids"] for x in items]
    keep = [i for i, s in enumerate(seqs) if len(s) > a.skip + 10]
    acts = np.zeros((len(keep), model.config.get_text_config().hidden_size), np.float32)
    pos = {k: j for j, k in enumerate(keep)}
    for idx in batches([seqs[i] for i in keep]):
        H = forward([seqs[keep[j]] for j in idx])
        for r, j in enumerate(idx):
            acts[j] = H[r, a.skip:len(seqs[keep[j]])].float().mean(0).cpu().numpy()
    np.savez(EO / "story_acts.npz", acts=acts.astype(np.float16), emotion=np.array([items[i]["emotion"] for i in keep]),
             topic=np.array([items[i]["topic_id"] for i in keep]))
    log(f"stories: {len(keep)}/{len(items)} stories harvested (>{a.skip + 10} tokens)")


def render_dialogue(d):
    msgs = ([{"role": "system", "content": d["system"]}] if d["system"] else []) + \
           [{"role": "user" if t["speaker"] == "person" else "assistant", "content": t["text"]} for t in d["turns"]]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def stage_neutral():
    rows = [json.loads(l) for l in open(DATA / "emotions/neutral.jsonl")]
    seqs = [render_dialogue(r) for r in rows]
    special = set(tok.all_special_ids)
    d = model.config.get_text_config().hidden_size
    S = torch.zeros(d, dtype=torch.float64, device=DEV); XX = torch.zeros(d, d, dtype=torch.float64, device=DEV); n = 0
    for idx in batches(seqs):
        H = forward([seqs[j] for j in idx])
        for r, j in enumerate(idx):
            m = torch.tensor([t not in special for t in seqs[j]], device=DEV); m[0] = False
            X = H[r, :len(seqs[j])][m].double()
            S += X.sum(0); XX += X.T @ X; n += len(X)
    mu = S / n
    C = XX / n - torch.outer(mu, mu)
    ev, evec = torch.linalg.eigh(C)
    ev, evec = ev.flip(0), evec.flip(1)
    k = int((torch.cumsum(ev, 0) / ev.sum() < 0.5).sum().item()) + 1
    np.savez(EO / "neutral_pca.npz", components=evec[:, :k].T.float().cpu().numpy(), explained=(ev[:k] / ev.sum()).cpu().numpy(),
             n_tokens=n, n_dialogues=len(seqs))
    log(f"neutral: {len(seqs)} dialogues, {n} tokens; top {k} PCs explain 50% of variance")


def stage_vectors():
    z = np.load(EO / "story_acts.npz"); P = np.load(EO / "neutral_pca.npz")["components"].astype(np.float32)
    emos = sorted(set(z["emotion"]))
    A = z["acts"].astype(np.float32)
    means = np.stack([A[z["emotion"] == e].mean(0) for e in emos])
    raw = means - means.mean(0)
    den = raw - (raw @ P.T) @ P
    # split-half reliability by topic (even/odd topic ids)
    tid = np.array([int(t[1:]) for t in z["topic"]])
    halves = []
    for h in (0, 1):
        m = np.stack([A[(z["emotion"] == e) & (tid % 2 == h)].mean(0) for e in emos]); halves.append(m - m.mean(0))
    rel = [float(halves[0][i] @ halves[1][i] / np.linalg.norm(halves[0][i]) / np.linalg.norm(halves[1][i])) for i in range(len(emos))]
    np.savez(EO / "emotion_vectors.npz", names=np.array(emos), raw=raw, denoised=den, split_half_cos=np.array(rel), layer=L)
    log(f"vectors: {len(emos)} emotions; split-half cos median {np.median(rel):.3f} min {np.min(rel):.3f}")


def stage_project():
    z = np.load(EO / "emotion_vectors.npz")
    V = np.concatenate([z["raw"], z["denoised"]])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    Vt = torch.tensor(V.T, device=DEV)
    (EO / "rollouts").mkdir(exist_ok=True)
    files = sorted((OUT / "cache").glob("*.npy"))
    t0 = time.time()
    for i, f in enumerate(files):
        dst = EO / "rollouts" / f"{f.stem}.npy"
        if dst.exists():
            continue
        X = torch.tensor(np.load(f), device=DEV).float()
        np.save(dst, (X @ Vt).cpu().numpy().astype(np.float32))
        if i % 25 == 0:
            log(f"project: {i + 1}/{len(files)} rollouts, {time.time() - t0:.0f}s")
    json.dump({"columns": [f"{n}__raw" for n in z["names"]] + [f"{n}__denoised" for n in z["names"]]},
              open(EO / "rollouts" / "columns.json", "w"))
    extra = DATA / "extra_directions.npz"
    if extra.exists():  # extra (unnormalised) directions, e.g. legit_shortcut plain/colloquial, same scale as session 1
        ez = np.load(extra); Et = torch.tensor(ez["V"].T, device=DEV, dtype=torch.float32)
        (OUT / "extra").mkdir(exist_ok=True)
        for f in files:
            dst = OUT / "extra" / f"{f.stem}.npy"
            if not dst.exists():
                np.save(dst, (torch.tensor(np.load(f), device=DEV).float() @ Et).cpu().numpy().astype(np.float32))
        json.dump({"columns": [str(x) for x in ez["names"]]}, open(OUT / "extra" / "columns.json", "w"))
        log(f"project: extra directions {list(ez['names'])} for {len(files)} rollouts")


try:
    for name, fn in (("stories", stage_stories), ("neutral", stage_neutral), ("vectors", stage_vectors), ("project", stage_project)):
        if (EO / f".{name}.done").exists():
            log(f"{name}: already done"); continue
        log(f"{name}: start"); fn(); (EO / f".{name}.done").write_text(time.strftime("%c")); log(f"{name}: done")
    log("SESSION2 COMPLETE")
except Exception:
    log("FAILED:\n" + traceback.format_exc()); (EO / "FAILED").write_text(traceback.format_exc()); sys.exit(1)
