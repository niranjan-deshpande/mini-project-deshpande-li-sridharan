"""CPU tests for gpu/lenses.py on the local Qwen3.5-0.8B (same class as Qwen3.8-27B: hybrid Gated-DeltaNet /
gated attention, (1 + w) RMSNorm, SiLU-gated MLP; 24 blocks, tied embeddings).

Fast suite (default): ~2-4 min on a laptop CPU. Timing benchmark: LENS_SLOW=1 pytest tests/test_lenses.py -s
Skips everything if the model directory is missing (override with QWEN_SMALL=/path).
"""
import math
import os
import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))
import lenses  # noqa: E402

MODEL_DIR = Path(os.environ.get("QWEN_SMALL", Path.home() / "hf_small" / "Qwen3.5-0.8B"))
slow = pytest.mark.skipif(not os.environ.get("LENS_SLOW"), reason="slow; set LENS_SLOW=1")

FALLBACK_TEXT = (
    " The history of the region is closely tied to the river , which provided water for agriculture and a route "
    "for trade between the coastal cities and the interior . During the nineteenth century , several new towns "
    "were founded along its banks , and the arrival of the railway in 1872 transformed the local economy . Farmers "
    "began to grow cash crops for distant markets , while merchants built warehouses and banks in the larger "
    "settlements . The population grew rapidly , and by the end of the century the area had become one of the "
    "most prosperous in the country . Schools , churches and a small university were established , and a local "
    "newspaper reported on politics , sport and the weather ."
)


@pytest.fixture(scope="module")
def mt():
    if not (MODEL_DIR / "config.json").exists():
        pytest.skip(f"model dir {MODEL_DIR} missing")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    # eager attention so torch.func.jvp (forward-mode AD) can serve as ground truth; fd_lens itself is agnostic
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    return model, tok


@pytest.fixture(scope="module")
def texts(mt):
    _, tok = mt
    try:
        t = lenses.wikitext_texts(4, 128, tok)
        if len(t) == 4:
            return t
    except Exception as e:  # offline: fall back to a fixed paragraph with varied prefixes
        print(f"wikitext unavailable ({e}); using fallback text")
    return [f"Section {i} . " + FALLBACK_TEXT for i in range(4)]


@pytest.fixture(scope="module")
def dirs(mt):
    model, tok = mt
    E = model.lm_head.weight.detach()
    words = [" Paris", " dog", " happy", " seven"]
    V = {w.strip(): E[tok.encode(w)[0]].clone() for w in words}
    g = torch.Generator().manual_seed(0)
    V["rand"] = torch.randn(E.shape[1], generator=g)
    return V


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.detach().double(), b.detach().double(), dim=0))


def overlap(a, b, key="top", k=20):
    return len({t[2] for t in a[key][:k]} & {t[2] for t in b[key][:k]}) / k


# ---------------------------------------------------------------------------------------------------- fast tests
def test_unembedding_row_decodes_to_token(mt, dirs):
    model, tok = mt
    res = lenses.logit_lens(model, tok, dirs, k=20)
    for w in ("Paris", "dog", "happy", "seven"):
        top = [t[0].strip().lower() for t in res[w]["top"][:3]]
        assert w.lower() in top, (w, res[w]["top"][:5])
        assert res[w]["top"][0][2] == tok.encode(" " + w)[0]  # tied embeddings: the row's own token ranks first


def test_readout_filter(mt, dirs):
    model, tok = mt
    mask = lenses.non_semantic_token_ids(tok, model.lm_head.weight.shape[0])
    assert 5000 < int(mask.sum()) < 8000  # 6,116 reported for the Qwen3.6 tokenizer
    res = lenses.logit_lens(model, tok, dirs, k=50)
    for r in res.values():
        assert all(any(c.isalnum() for c in t[0]) for t in r["top"] + r["bottom"])
    unf = lenses.logit_lens(model, tok, dirs, k=50, filter_tokens=False)
    assert any(not any(c.isalnum() for c in t[0]) for r in unf.values() for t in r["top"] + r["bottom"])


def test_matrix_lens_identity_equals_logit_lens(mt, dirs):
    model, tok = mt
    a = lenses.logit_lens(model, tok, dirs)
    b = lenses.matrix_lens(torch.eye(model.lm_head.weight.shape[1]), model, tok, dirs)
    for n in dirs:
        assert [t[2] for t in a[n]["top"]] == [t[2] for t in b[n]["top"]]
        assert [t[2] for t in a[n]["bottom"]] == [t[2] for t in b[n]["bottom"]]
        assert torch.allclose(a[n]["vec"], b[n]["vec"])


def test_fd_last_block_is_logit_lens(mt, dirs, texts):
    """layer = last block: J = I exactly (no blocks after it), for both modes."""
    model, tok = mt
    last = lenses._parts(model).n_layers - 1
    ll = lenses.logit_lens(model, tok, dirs)
    for mode in ("j", "r"):
        fd = lenses.fd_lens(model, tok, dirs, last, texts[:2], mode=mode)
        for n, v in dirs.items():
            assert cos(fd[n]["vec"], v) > 0.9999
            assert overlap(fd[n], ll[n]) >= 0.95


def test_fd_penultimate_block_close_to_logit_lens(mt, dirs, texts):
    """One block of transport: still close to the logit lens, but not identical."""
    model, tok = mt
    L = lenses._parts(model).n_layers - 2
    ll = lenses.logit_lens(model, tok, dirs)
    fd = lenses.fd_lens(model, tok, dirs, L, texts[:2], mode="j")
    agr = lenses.lens_agreement(ll, fd, model=model)
    tok_dirs = [n for n in dirs if n != "rand"]
    assert min(agr[n]["top_overlap"] for n in tok_dirs) >= 0.5
    assert min(agr[n]["cos_logits"] for n in tok_dirs) > 0.3  # observed 0.49-0.82 (full-vocab logits move more)
    assert max(agr[n]["cos_vec"] for n in dirs) < 0.999
    s = agr["_summary"]
    assert s["median_cos_logits"] > 0.6 and s["baseline_cos_logits"] < 0.1


@pytest.mark.parametrize("mode", ["j", "r"])
def test_fd_central_difference_linear(mt, dirs, texts, mode):
    """Halving eps leaves the readout unchanged: we are in the linear regime."""
    model, tok = mt
    sub = {n: dirs[n] for n in ("Paris", "rand")}
    a = lenses.fd_lens(model, tok, sub, 16, texts[:2], mode=mode, eps_frac=0.02)
    b = lenses.fd_lens(model, tok, sub, 16, texts[:2], mode=mode, eps_frac=0.01)
    for n in sub:
        assert cos(a[n]["vec"], b[n]["vec"]) > 0.99
        assert overlap(a[n], b[n]) >= 0.8


def _jvp_reference(model, tok, v, layer, texts, lrp: bool):
    """Ground truth J_layer v (reference estimator) by forward-mode AD. With lrp=True the forward graph carries the
    jpp_lens 'rlens' detach rules (copied from workspace_lens/lrp/lrp.py), so the JVP is the LRP-modified one."""
    from types import MethodType
    p = lenses._parts(model)
    ids = lenses._tokenize(tok, texts, 128)
    T = ids.shape[1]
    keep = torch.zeros(T, dtype=torch.bool)
    keep[16:T - 1] = True
    with torch.no_grad():
        h0 = lenses._run_prefix(p, ids, layer)

    def norm_fwd(self, x):
        x_f = x.float()
        scale = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps).detach()
        return (x_f * scale * (1.0 + self.weight.float())).type_as(x)

    def mlp_fwd(self, x):
        z, up = self.gate_proj(x), self.up_proj(x)
        prod = z * torch.sigmoid(z).detach() * up
        prod = 0.5 * prod + 0.5 * prod.detach()
        return self.down_proj(prod)

    patched = []
    if lrp:
        for i in range(layer + 1, p.n_layers):
            for m, f in ((p.layers[i].input_layernorm, norm_fwd), (p.layers[i].post_attention_layernorm, norm_fwd),
                         (p.layers[i].mlp, mlp_fwd)):
                object.__setattr__(m, "forward", MethodType(f, m))
                patched.append(m)
    try:
        vh = (v / v.norm())[None, None, :] * keep[None, :, None].float()
        _, jv = torch.func.jvp(lambda h: lenses._run_suffix(p, h, layer)[:, keep].sum(dim=(0, 1)),
                               (h0,), (vh.expand_as(h0).contiguous(),))
    finally:
        for m in patched:
            m.__dict__.pop("forward", None)
    return jv / (int(keep.sum()) * len(ids))


@pytest.mark.parametrize("mode", ["j", "r"])
def test_fd_matches_forward_mode_ad(mt, dirs, texts, mode):
    """FD estimate == JVP through the stock graph (mode j) / the jpp_lens LRP-detached graph (mode r)."""
    model, tok = mt
    L = 18
    v = dirs["rand"]
    ref = _jvp_reference(model, tok, v, L, texts[:2], lrp=(mode == "r"))
    fd = lenses.fd_lens(model, tok, {"v": v}, L, texts[:2], mode=mode)["v"]["vec"]
    assert cos(fd, ref) > 0.9995
    assert (fd - ref).norm() / ref.norm() < 0.02  # same absolute normalisation as the reference estimator


def test_r_lens_differs_from_j_but_correlates(mt, dirs, texts):
    model, tok = mt
    sub = {n: dirs[n] for n in ("Paris", "happy", "rand")}
    j = lenses.fd_lens(model, tok, sub, 12, texts[:2], mode="j")
    r = lenses.fd_lens(model, tok, sub, 12, texts[:2], mode="r")
    for n in sub:
        c = cos(j[n]["vec"], r[n]["vec"])
        assert 0.2 < c < 0.995, (n, c)
    for n in ("Paris", "happy"):  # both still find the token at the top
        assert overlap(j[n], r[n], k=10) >= 0.3


def test_r_patching_preserves_clean_forward(mt, texts):
    model, tok = mt
    p = lenses._parts(model)
    L = 14
    ids = lenses._tokenize(tok, texts[:2], 128)
    with torch.no_grad():
        h0 = lenses._run_prefix(p, ids, L)
        stock = lenses._run_suffix(p, h0, L)
        state = lenses._LrpState()
        with lenses._lrp_patched(p, L, state):
            rec = lenses._run_suffix(p, h0, L)  # record pass: stock forward
            state.mode = "replay"
            rep = lenses._run_suffix(p, h0, L)  # surrogate at the clean point
            rep2 = lenses._run_suffix(p, torch.cat([h0, h0]), L)  # replicated batch
        after = lenses._run_suffix(p, h0, L)  # patches removed
    assert torch.equal(rec, stock)
    assert torch.equal(after, stock)
    rel = float((rep - stock).norm() / stock.norm())
    assert rel < 1e-5, rel
    assert torch.allclose(rep2[:2], rep2[2:], rtol=0, atol=1e-4) and torch.allclose(rep2[:2], rep, atol=1e-4)
    assert all("forward" not in m.__dict__ for m in model.modules())


def test_load_jpp_layer_fake_files(tmp_path):
    d = 8
    Js = {l: torch.randn(d, d) for l in (8, 16, 40)}
    # jpp_lens BaseLens.save format (fp16 on disk)
    f1 = tmp_path / "lens.pt"
    torch.save({"parameters": {"jacobians": {l: J.half() for l, J in Js.items()}}, "source_layers": [8, 16, 40],
                "config": {"lens_type": "jacobian", "relative_end_transport_layer": -1, "lrp_mode": "rlens"}}, f1)
    J = lenses.load_jpp_layer(str(f1), 40)
    assert J.dtype == torch.float32 and torch.allclose(J, Js[40].half().float())
    J2, meta = lenses.load_jpp_layer(str(tmp_path), 16, filename="lens.pt", return_meta=True)
    assert torch.allclose(J2, Js[16].half().float()) and meta["source_layers"] == [8, 16, 40]
    # legacy jacobian-lens / R-lens format
    f2 = tmp_path / "legacy.pt"
    torch.save({"J": Js, "n_prompts": 3, "source_layers": [8, 16, 40], "d_model": d}, f2)
    assert torch.equal(lenses.load_jpp_layer(str(f2), 8), Js[8])
    with pytest.raises(KeyError):
        lenses.load_jpp_layer(str(f2), 41)
    with pytest.raises(ValueError):
        torch.save({"foo": 1}, tmp_path / "bad.pt")
        lenses.load_jpp_layer(str(tmp_path / "bad.pt"), 8)


def test_matrix_lens_with_loaded_file(mt, dirs, tmp_path):
    model, tok = mt
    d = model.lm_head.weight.shape[1]
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(1)))
    torch.save({"parameters": {"jacobians": {5: Q}}, "source_layers": [5], "config": {}}, tmp_path / "lens.pt")
    J = lenses.load_jpp_layer(str(tmp_path / "lens.pt"), 5)
    res = lenses.matrix_lens(J, model, tok, dirs)
    for n, v in dirs.items():
        assert torch.allclose(res[n]["vec"], Q @ v, atol=1e-4)


def test_lens_agreement(mt, dirs):
    model, tok = mt
    a = lenses.logit_lens(model, tok, dirs)
    agr = lenses.lens_agreement(a, a, model=model)
    for n in dirs:
        assert agr[n]["top_overlap"] == 1.0 and abs(agr[n]["cos_logits"] - 1) < 1e-6 and abs(agr[n]["cos_vec"] - 1) < 1e-6
    s = agr["_summary"]
    assert s["baseline_top_overlap"] < 0.2 and s["baseline_cos_logits"] < 0.5
    neg = lenses.logit_lens(model, tok, {n: -v for n, v in dirs.items()})
    agr2 = lenses.lens_agreement(a, neg)
    assert all(agr2[n]["cos_vec"] < -0.999 for n in dirs)


def test_wikitext_texts_deterministic(mt):
    _, tok = mt
    try:
        a = lenses.wikitext_texts(6, 128, tok)
    except Exception as e:
        pytest.skip(f"wikitext unavailable: {e}")
    b = lenses.wikitext_texts(6, 128, tok)
    assert a == b and len(a) == 6
    assert all(len(tok(t).input_ids) >= 128 and len(t.strip()) >= 600 for t in a)
    assert lenses.wikitext_texts(3, 128, tok) == a[:3]


# ---------------------------------------------------------------------------------------------------- timing
@slow
def test_timing_and_extrapolation(mt, texts):
    """Per-direction CPU cost at a deep-suffix layer, and an FLOP-scaled H100 estimate for the real run."""
    model, tok = mt
    p = lenses._parts(model)
    L = 8  # 15 suffix blocks
    D = 4
    V = torch.randn(D, p.d_model, generator=torch.Generator().manual_seed(3))
    rows = {}
    for mode in ("j", "r"):
        _, st = lenses.fd_lens(model, tok, V, L, texts[:4], mode=mode, batch_size=4, dir_batch=2,
                               return_stats=True)
        rows[mode] = st
    blk_params = sum(x.numel() for x in p.layers.parameters()) / p.n_layers
    n_suffix = p.n_layers - 1 - L
    tok_passes = 2 * 4 * 128 * D  # signs x texts x tokens x directions
    flop = 2 * blk_params * n_suffix * tok_passes
    for mode, st in rows.items():
        print(f"\n[{mode}] {st['seconds']:.1f}s for {D} dirs x 4 texts x 128 tok at layer {L} "
              f"({n_suffix} suffix blocks): {st['seconds_per_direction']:.2f} s/dir, "
              f"~{flop / st['seconds'] / 1e9:.0f} GFLOP/s effective")
    # Qwen3.8-27B: ~0.38e9 params/block (27e9 - 2 x 248320 x 5120 untied embed/unembed, / 64), layer 40 -> 23 blocks
    big = 2 * 0.38e9 * 23 * (2 * 64 * 128) * 45
    for mfu in (0.15, 0.35):
        print(f"H100 estimate, 45 dirs x 64 x 128 tok at layer 40, {mfu:.0%} of 989 TFLOP/s bf16: "
              f"{big / (mfu * 989e12):.0f} s per mode")
