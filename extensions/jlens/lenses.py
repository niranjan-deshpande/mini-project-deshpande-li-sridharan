"""Lenses for decoding residual-stream directions: logit lens, native finite-difference J-/R-lens, and a
borrowed fixed-matrix lens (J++). See notes/jpp_lens.md for the method definitions.

Conventions (match anthropics/jacobian-lens and safety-research/jpp_lens):
  * "layer L" is the output of decoder block ``layers[L]`` (HF ``hidden_states[L + 1]``).
  * Decoding is ``unembed(x) = lm_head(final_norm(x))`` with the model's own final norm module. The final norm is
    applied to the transported vector ``J v`` itself (scale-invariant, so the FD scale never matters).
  * The J-lens estimator is ``J_l = mean_prompts mean_{s in S} sum_{t in S, t >= s} dh_final[t] / dh_l[s]`` with
    ``S = [skip_first, T - 1)`` (cotangent at every kept target position, rows read at kept source positions,
    mean over source positions). ``h_final`` is the output of the last block (before the final norm).
  * Readout filtering: tokens whose decoded string has no letter or digit (``str.isalnum``) are excluded from the
    top/bottom ranking; ids are scanned over the lm_head width.

Native FD lens (``fd_lens``): because blocks are causal, adding ``eps * v`` at every kept source position of the
layer-L output and summing the change in ``h_final`` over kept target positions gives
``sum_s sum_{t >= s} J_ts v * eps`` to first order, i.e. ``|S| * eps * J_l v`` for one prompt. Central differences
cancel the even-order terms. The R-lens variant (mode="r") runs the perturbed passes through a surrogate network
whose Jacobian at the clean point is the LRP-modified one (R-lens / J++ "rlens" rules):
  * LN-rule: every residual-stream ``Qwen3_5RMSNorm`` (input_layernorm, post_attention_layernorm) of blocks
    L+1..last uses the clean pass's per-position ``rsqrt(mean(x^2) + eps)`` (so it is linear in x).
  * Identity rule + half rule on the gated MLP: ``act(z) * u`` is replaced by the affine map
    ``0.5 * (z * sigmoid(z0) * u0 + u * sigmoid(z0) * z0)``, which equals ``silu(z0) * u0`` at the clean point and
    whose derivative is ``0.5 * (sigmoid(z0) u0 dz + sigmoid(z0) z0 du)`` = the reference's detached-sigmoid, halved
    gradient.
  * Attention, q/k norms, the Gated-DeltaNet gated norm (``Qwen3_5RMSNormGated``, not matched by the reference
    either) and everything else keep their true (plain-gradient) behaviour, which the FD propagates exactly.
The final norm is outside the Jacobian (target = last block output), so it is never patched.
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.parse
import urllib.request
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import torch
from torch import nn

__all__ = ["logit_lens", "fd_lens", "matrix_lens", "load_jpp_layer", "wikitext_texts", "lens_agreement",
           "non_semantic_token_ids", "unembed", "format_readouts"]


# ------------------------------------------------------------------------------------------------ model plumbing
@dataclass
class _Parts:
    text_model: nn.Module
    layers: nn.ModuleList
    final_norm: nn.Module
    lm_head: nn.Module
    n_layers: int
    d_model: int


def _parts(model: nn.Module) -> _Parts:
    """Locate the text decoder inside a *ForCausalLM / *ForConditionalGeneration (or a bare text model)."""
    for path in ("model.language_model", "model", "language_model", ""):
        obj = model
        try:
            for a in filter(None, path.split(".")):
                obj = getattr(obj, a)
        except AttributeError:
            continue
        if all(hasattr(obj, a) for a in ("layers", "norm", "embed_tokens")):
            lm_head = getattr(model, "lm_head", None)
            if lm_head is None:
                raise ValueError(f"{type(model).__name__} has no lm_head")
            return _Parts(obj, obj.layers, obj.norm, lm_head, len(obj.layers), obj.embed_tokens.weight.shape[1])
    raise ValueError(f"could not locate the text decoder inside {type(model).__name__}")


class _Stop(Exception):
    pass


class _PassThrough(nn.Module):
    """Stands in for a decoder block that must not run (its output is already known)."""

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


@contextmanager
def _skip_blocks(p: _Parts, upto: int):
    """Temporarily replace blocks 0..upto with pass-throughs so ``text_model(inputs_embeds=h_upto)`` runs only
    blocks upto+1..last, while HF still builds the masks / rotary embeddings / position ids for the actual batch."""
    saved = [p.layers[i] for i in range(upto + 1)]
    try:
        for i in range(upto + 1):
            p.layers[i] = _PassThrough()
        yield
    finally:
        for i, m in enumerate(saved):
            p.layers[i] = m


def _run_prefix(p: _Parts, ids: torch.Tensor, layer: int) -> torch.Tensor:
    """Output of block ``layer`` for input ids [B, T] (the forward is aborted right after that block)."""
    out = {}

    def hook(mod, inp, o):
        out["h"] = o[0] if isinstance(o, tuple) else o
        raise _Stop

    h = p.layers[layer].register_forward_hook(hook)
    try:
        p.text_model(input_ids=ids, use_cache=False)
    except _Stop:
        pass
    finally:
        h.remove()
    return out["h"]


def _run_suffix(p: _Parts, h: torch.Tensor, layer: int) -> torch.Tensor:
    """Run blocks layer+1..last on h [B, T, d] (the layer-``layer`` output); returns the last block's output
    (the input of the final norm, which is not run)."""
    if layer == p.n_layers - 1:
        return h
    out = {}

    def pre_hook(mod, args):
        out["h"] = args[0]
        raise _Stop

    hk = p.final_norm.register_forward_pre_hook(pre_hook)
    try:
        with _skip_blocks(p, layer):
            p.text_model(inputs_embeds=h, use_cache=False)
    except _Stop:
        pass
    finally:
        hk.remove()
    return out["h"]


def _suffix_dtype(p: _Parts, layer: int) -> torch.dtype:
    blk = p.layers[min(layer + 1, p.n_layers - 1)]
    return next(blk.parameters()).dtype


@torch.no_grad()
def unembed(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """``lm_head(final_norm(X))`` in the lm_head's dtype/device; returns fp32 logits [..., V]."""
    p = _parts(model)
    w = p.lm_head.weight
    return p.lm_head(p.final_norm(X.to(device=w.device, dtype=w.dtype))).float()


# ------------------------------------------------------------------------------------------------ readout
_NONSEM_CACHE: dict = {}


def non_semantic_token_ids(tok, vocab_size: int) -> torch.Tensor:
    """Bool mask [vocab_size]: True for ids whose decoded string has no letter or digit (incl. lm_head padding rows
    that decode to ''). Same definition as jpp_lens ``lens_evals.eval_utils.is_non_semantic_token``."""
    key = (getattr(tok, "name_or_path", id(tok)), len(tok), vocab_size)
    if key not in _NONSEM_CACHE:
        strs = tok.batch_decode([[i] for i in range(vocab_size)])
        _NONSEM_CACHE[key] = torch.tensor([not any(c.isalnum() for c in s) for s in strs], dtype=torch.bool)
    return _NONSEM_CACHE[key]


def _as_matrix(V) -> tuple[list[str], torch.Tensor]:
    if isinstance(V, dict):
        names = [str(n) for n in V]
        X = torch.stack([torch.as_tensor(V[n]).float().reshape(-1) for n in V])
    else:
        X = torch.as_tensor(V).float()
        X = X[None] if X.ndim == 1 else X
        names = [str(i) for i in range(len(X))]
    return names, X


def _readouts(model, tok, names, vecs: torch.Tensor, k: int, filter_tokens: bool) -> dict:
    """Decode transported vectors [D, d] -> {name: {"top": [[str, logit, id]...], "bottom": [...], "vec": [d]}}."""
    p = _parts(model)
    V = p.lm_head.weight.shape[0]
    mask = non_semantic_token_ids(tok, V) if filter_tokens else torch.zeros(V, dtype=torch.bool)
    res = {}
    for i, n in enumerate(names):
        lg = unembed(model, vecs[i:i + 1])[0].cpu()
        hi = lg.masked_fill(mask, -float("inf")).topk(k)
        lo = (-lg).masked_fill(mask, -float("inf")).topk(k)
        res[n] = {
            "top": [[tok.decode([j]), float(lg[j]), int(j)] for j in hi.indices.tolist()],
            "bottom": [[tok.decode([j]), float(lg[j]), int(j)] for j in lo.indices.tolist()],
            "vec": vecs[i].detach().float().cpu(),
        }
    return res


def format_readouts(res: dict, n: int = 10) -> str:
    lines = []
    for name, r in res.items():
        lines.append(f"{name}\n  + " + " ".join(repr(t[0]) for t in r["top"][:n])
                     + "\n  - " + " ".join(repr(t[0]) for t in r["bottom"][:n]))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ lenses
@torch.no_grad()
def logit_lens(model, tok, V, k: int = 20, filter_tokens: bool = True) -> dict:
    """Top-k / bottom-k tokens of ``lm_head(final_norm(v))`` for each direction (dict name -> [d], or [D, d])."""
    names, X = _as_matrix(V)
    return _readouts(model, tok, names, X, k, filter_tokens)


@torch.no_grad()
def matrix_lens(J: torch.Tensor, model, tok, V, k: int = 20, filter_tokens: bool = True) -> dict:
    """Fixed linear lens: decode ``unembed(J @ v)`` for a [d, d] map J (rows = final-layer basis, like the
    jacobian-lens / jpp_lens ``transport``: ``residual @ J.T``). ``J = I`` is the logit lens."""
    names, X = _as_matrix(V)
    J = J.float()
    return _readouts(model, tok, names, X.to(J.device) @ J.T, k, filter_tokens)


def _tokenize(tok, texts, n_tokens: int) -> torch.Tensor:
    if isinstance(texts, torch.Tensor):
        return texts[:, :n_tokens].long()
    rows = []
    for t in texts:
        ids = tok(t, truncation=True, max_length=n_tokens, add_special_tokens=True).input_ids
        if len(ids) == n_tokens:
            rows.append(ids)
    if len(rows) < len(texts):
        warnings.warn(f"fd_lens: dropped {len(texts) - len(rows)}/{len(texts)} texts shorter than {n_tokens} tokens")
    if not rows:
        raise ValueError("no text has n_tokens tokens")
    return torch.tensor(rows, dtype=torch.long)


class _LrpState:
    """Per-module caches for the R-lens surrogate. mode: 'record' (stock forward, cache clean factors) or 'replay'
    (surrogate forward; the batch is R replicas of the cached B sequences, replica-major)."""

    def __init__(self, cache_dtype=torch.float32):
        self.mode = "record"
        self.cache: dict[int, tuple] = {}
        self.cache_dtype = cache_dtype


def _norm_forward(state: _LrpState):
    def forward(self, x):
        xf = x.float()
        if state.mode == "record":
            scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
            state.cache[id(self)] = (scale,)
        else:
            (scale,) = state.cache[id(self)]
            B = scale.shape[0]
            scale = scale.repeat(xf.shape[0] // B, *([1] * (xf.ndim - 1)))
        # identical op order to Qwen3_5RMSNorm: (x * rsqrt(.)) * (1 + w), then type_as(x)
        return (xf * scale * (1.0 + self.weight.float())).type_as(x)
    return forward


def _mlp_forward(state: _LrpState):
    def forward(self, x):
        z, u = self.gate_proj(x), self.up_proj(x)
        if state.mode == "record":
            s = torch.sigmoid(z.float())
            state.cache[id(self)] = ((s * u.float()).to(state.cache_dtype), (s * z.float()).to(state.cache_dtype))
            return self.down_proj(self.act_fn(z) * u)
        a, b = state.cache[id(self)]
        B = a.shape[0]
        R = z.shape[0] // B
        zr = z.float().view(R, B, *z.shape[1:])
        ur = u.float().view(R, B, *u.shape[1:])
        prod = 0.5 * (zr * a.float()[None] + ur * b.float()[None])
        return self.down_proj(prod.view_as(z).to(z.dtype))
    return forward


@contextmanager
def _lrp_patched(p: _Parts, layer: int, state: _LrpState):
    """Instance-patch the residual-stream RMSNorms and gated MLPs of blocks layer+1..last (class untouched)."""
    saved = []
    try:
        for i in range(layer + 1, p.n_layers):
            blk = p.layers[i]
            for name in ("input_layernorm", "post_attention_layernorm"):
                m = getattr(blk, name)
                # the surrogate reproduces the (1 + w) zero-centred Qwen3.5 RMSNorm forward exactly
                if type(m).__name__ not in ("Qwen3_5RMSNorm", "Qwen3_5MoeRMSNorm"):
                    raise RuntimeError(f"no LN-rule surrogate for {type(m).__name__} at block {i}.{name}")
                saved.append(m)
                object.__setattr__(m, "forward", MethodType(_norm_forward(state), m))
            mlp = blk.mlp
            if not all(hasattr(mlp, a) for a in ("gate_proj", "up_proj", "down_proj", "act_fn")) or \
                    type(mlp.act_fn).__name__ not in ("SiLU", "SiLUActivation"):
                raise RuntimeError(f"block {i}.mlp is not a SiLU-gated MLP ({type(mlp).__name__})")
            saved.append(mlp)
            object.__setattr__(mlp, "forward", MethodType(_mlp_forward(state), mlp))
        yield
    finally:
        for m in saved:
            m.__dict__.pop("forward", None)


@torch.no_grad()
def fd_lens(model, tok, V, layer: int, texts, mode: str = "j", eps_frac: float = 0.02, skip_first: int = 16,
            batch_size: int = 8, dir_batch: int = 4, n_tokens: int = 128, k: int = 20, filter_tokens: bool = True,
            eps: float | None = None, return_stats: bool = False):
    """Native J-lens (mode="j") or R-lens (mode="r") readout of each direction at ``layer`` by central finite
    differences. Returns ``{name: {"top", "bottom", "vec"}}`` where ``vec`` is the estimate of ``J_layer v_hat``
    (unit-norm v; same normalisation as the reference estimator, mean over prompts and source positions).

    texts: list of strings (tokenized, truncated to ``n_tokens``; shorter ones dropped) or a LongTensor [N, T].
    eps: absolute perturbation norm; default ``eps_frac * median ||h_layer||`` over kept positions of the first batch.
    batch_size texts x dir_batch directions x 2 signs sequences are run through the suffix per forward.
    """
    if mode not in ("j", "r"):
        raise ValueError(f"mode must be 'j' or 'r', got {mode!r}")
    p = _parts(model)
    if not 0 <= layer < p.n_layers:
        raise ValueError(f"layer {layer} out of range for {p.n_layers} blocks")
    t0 = time.time()
    names, X = _as_matrix(V)
    D = len(names)
    vhat = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ids_all = _tokenize(tok, texts, n_tokens)
    N, T = ids_all.shape
    if T <= skip_first + 1:
        raise ValueError(f"sequence length {T} leaves no positions after skip_first={skip_first}")
    dev = p.text_model.embed_tokens.weight.device
    sdt = _suffix_dtype(p, layer)
    keep = torch.zeros(T, dtype=torch.bool)
    keep[skip_first:T - 1] = True
    n_src = int(keep.sum())
    acc = torch.zeros(D, p.d_model, dtype=torch.float64)
    state = _LrpState()
    for b0 in range(0, N, batch_size):
        ids = ids_all[b0:b0 + batch_size].to(dev)
        B = len(ids)
        h0 = _run_prefix(p, ids, layer).to(sdt)
        # device_map="auto" shards the model: h0 lives on the transport block's device and the suffix output on
        # the final-norm's device, neither necessarily the embedding device. Build every mask/vector on the
        # device of the tensor it multiplies/indexes, not on a single pinned `dev`.
        hdev = h0.device
        keep_h = keep.to(hdev)
        vhat_h = vhat.to(hdev)
        if eps is None:
            eps = eps_frac * float(h0.float()[:, keep_h].norm(dim=-1).median())
        ctx = _lrp_patched(p, layer, state) if mode == "r" else _nullctx()
        with ctx:
            if mode == "r":
                state.mode, state.cache = "record", {}
                _run_suffix(p, h0, layer)
                state.mode = "replay"
            for d0 in range(0, D, dir_batch):
                Dc = min(dir_batch, D - d0)
                pert = eps * vhat_h[d0:d0 + Dc, None, None, :] * keep_h[None, None, :, None]  # [Dc, 1, T, d]
                H = torch.stack([h0.float()[None] + pert, h0.float()[None] - pert])  # [2, Dc, B, T, d]
                out = _run_suffix(p, H.view(2 * Dc * B, T, -1).to(sdt), layer).float().view(2, Dc, B, T, -1)
                keep_o = keep.to(out.device)
                delta = (out[0] - out[1])[:, :, keep_o].sum(dim=(1, 2))  # sum over texts and kept targets
                acc[d0:d0 + Dc] += delta.double().cpu()
        state.cache = {}
    Jv = (acc / (2 * eps * n_src * N)).float()
    res = _readouts(model, tok, names, Jv, k, filter_tokens)
    if return_stats:
        dt = time.time() - t0
        stats = {"mode": mode, "layer": layer, "eps": eps, "n_texts": N, "n_tokens": T, "n_source_positions": n_src,
                 "n_directions": D, "seconds": dt, "seconds_per_direction": dt / D, "suffix_dtype": str(sdt)}
        return res, stats
    return res


@contextmanager
def _nullctx():
    yield


# ------------------------------------------------------------------------------------------------ borrowed J++ lens
def load_jpp_layer(path_or_repo: str, layer: int, filename: str = "qwen3.6-27b/lens.pt", revision: str | None = None,
                   return_meta: bool = False):
    """Load one layer's [d, d] fp32 transport matrix from a jpp_lens / jacobian-lens file.

    ``path_or_repo``: a lens file, a directory containing ``filename``, or a Hub repo id (e.g. "koayon/jpp-lenses";
    WARNING: the Qwen3.6-27B file is ~6.6 GB and is downloaded on first use).
    Formats: jpp_lens ``BaseLens.save`` -> {"parameters": {"jacobians": {layer: J}}, "source_layers", "config"};
    legacy jacobian-lens / R-lens files -> {"J": {layer: J}, "n_prompts", "source_layers", "d_model"[, "provenance"]}.
    ``J`` is [F, N] (target basis x source basis); apply as ``h @ J.T``. The file is mmap'd, so only this layer's
    ~105 MB is read. With ``return_meta`` also returns (source_layers, config/provenance dict).
    """
    if os.path.isfile(path_or_repo):
        path = path_or_repo
    elif os.path.isdir(path_or_repo):
        path = os.path.join(path_or_repo, filename)
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(path_or_repo, filename, revision=revision)
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError:  # legacy (non-zip) serialisation cannot be mmap'd
        ck = torch.load(path, map_location="cpu", weights_only=True)
    if "parameters" in ck:
        Js = ck["parameters"]["jacobians"]
        meta = {"source_layers": list(ck.get("source_layers", sorted(Js))), "config": ck.get("config", {})}
    elif "J" in ck:
        Js = ck["J"]
        meta = {"source_layers": list(ck.get("source_layers", sorted(Js))),
                "config": {"n_prompts": ck.get("n_prompts"), "d_model": ck.get("d_model"),
                           "provenance": ck.get("provenance")}}
    else:
        raise ValueError(f"{path} is not a lens file (keys {sorted(ck)!r})")
    key = layer if layer in Js else str(layer) if str(layer) in Js else None
    if key is None:
        raise KeyError(f"layer {layer} not in lens file (has {sorted(Js)[:5]}...{sorted(Js)[-3:]})")
    J = Js[key].float().clone()
    cfg = meta["config"] if isinstance(meta["config"], dict) else {}
    rel = cfg.get("relative_end_transport_layer")
    if rel not in (None, -1):
        warnings.warn(f"lens targets relative_end_transport_layer={rel}, not the final block")
    return (J, meta) if return_meta else J


# ------------------------------------------------------------------------------------------------ data
_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def _wikitext_records(n_needed: int, min_chars: int, cache: Path) -> list[str]:
    """First records (train order) of WikiText-103-raw with >= min_chars stripped characters, cached on disk."""
    got, offset = [], 0
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("min_chars") == min_chars:
            got, offset = c["texts"], c["offset"]
    if len(got) >= n_needed:
        return got[:n_needed]
    try:
        import datasets  # noqa: F401
        have_datasets = True
    except ImportError:
        have_datasets = False
    if have_datasets:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        got, offset = [], 0
        for rec in ds:
            offset += 1
            if len(rec["text"].strip()) >= min_chars:
                got.append(rec["text"])
                if len(got) >= n_needed:
                    break
    else:  # HF datasets-server (free, no auth), same row order as the streamed parquet
        while len(got) < n_needed:
            q = urllib.parse.urlencode(dict(dataset="Salesforce/wikitext", config="wikitext-103-raw-v1",
                                            split="train", offset=offset, length=100))
            for attempt in range(4):
                try:
                    rows = json.load(urllib.request.urlopen(f"{_ROWS_URL}?{q}", timeout=60))["rows"]
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            if not rows:
                break
            for r in rows:
                if len(r["row"]["text"].strip()) >= min_chars:
                    got.append(r["row"]["text"])
            offset += len(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"min_chars": min_chars, "offset": offset, "texts": got}))
    return got[:n_needed]


def wikitext_texts(n: int = 64, n_tokens: int = 128, tok=None, min_chars: int = 600,
                   cache_dir: str | os.PathLike | None = None) -> list[str]:
    """Deterministic WikiText-103 (raw, train) sample as in jacobian-lens / jpp_lens ``load_wikitext_prompts``:
    the first records with >= ``min_chars`` characters, in dataset order. With ``tok``, records that tokenize to
    fewer than ``n_tokens`` tokens are skipped (fd_lens needs equal-length sequences), so exactly ``n`` usable texts
    are returned. Uses ``datasets`` streaming if installed, else the HF datasets-server rows API; cached as JSON."""
    cache = Path(cache_dir or Path.home() / ".cache" / "decomp_lenses") / f"wikitext103_train_min{min_chars}.json"
    want = n
    while True:
        recs = _wikitext_records(want, min_chars, cache)
        if tok is None:
            return recs[:n]
        ok = [t for t in recs if len(tok(t, add_special_tokens=True).input_ids) >= n_tokens]
        if len(ok) >= n or len(recs) < want:
            if len(ok) < n:
                warnings.warn(f"wikitext_texts: only {len(ok)} texts with >= {n_tokens} tokens")
            return ok[:n]
        want = int(want * 1.25) + (n - len(ok)) + 1


# ------------------------------------------------------------------------------------------------ agreement
def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=-1))


def lens_agreement(a: dict, b: dict, k: int = 20, model=None) -> dict:
    """Compare two readout dicts (output of any lens here) direction by direction.

    Per direction: ``cos_vec`` (cosine of the transported vectors), ``cos_logits`` (centred cosine of the full
    unembedded logit vectors; needs ``model``), ``top_overlap`` / ``bottom_overlap`` (|top-k a & top-k b| / k on the
    stored, possibly filtered, rankings). ``_summary`` holds medians plus a mismatched-pair baseline (a[i] vs b[j],
    i != j) for cos_vec / cos_logits / top_overlap, the reference level for "agreement".
    """
    names = [n for n in a if n in b and not n.startswith("_")]
    out = {}
    logits = {}
    if model is not None:
        for tag, r in (("a", a), ("b", b)):
            L = unembed(model, torch.stack([torch.as_tensor(r[n]["vec"]).float() for n in names])).cpu()
            logits[tag] = L - L.mean(-1, keepdim=True)
    va = torch.stack([torch.as_tensor(a[n]["vec"]).float() for n in names])
    vb = torch.stack([torch.as_tensor(b[n]["vec"]).float() for n in names])

    def ov(x, y, key, kk):
        return len({t[2] for t in x[key][:kk]} & {t[2] for t in y[key][:kk]}) / kk

    for i, n in enumerate(names):
        kk = min(k, len(a[n]["top"]), len(b[n]["top"]))
        m = {"cos_vec": _cos(va[i], vb[i]), "top_overlap": ov(a[n], b[n], "top", kk),
             "bottom_overlap": ov(a[n], b[n], "bottom", kk)}
        if logits:
            m["cos_logits"] = _cos(logits["a"][i], logits["b"][i])
        out[n] = m
    if names:
        import statistics as st
        summ = {f"median_{key}": st.median(out[n][key] for n in names) for key in out[names[0]]}
        if len(names) > 1:
            pairs = [(i, j) for i in range(len(names)) for j in range(len(names)) if i != j]
            kk = min(k, len(a[names[0]]["top"]), len(b[names[0]]["top"]))
            summ["baseline_cos_vec"] = st.median(_cos(va[i], vb[j]) for i, j in pairs)
            summ["baseline_top_overlap"] = st.median(ov(a[names[i]], b[names[j]], "top", kk) for i, j in pairs)
            if logits:
                summ["baseline_cos_logits"] = st.median(_cos(logits["a"][i], logits["b"][j]) for i, j in pairs)
        out["_summary"] = summ
    return out
