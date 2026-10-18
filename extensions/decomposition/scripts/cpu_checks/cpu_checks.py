"""CPU checks on a tiny random Qwen3_5ForConditionalGeneration (torch fallback kernels)."""
import copy, os, sys, json, tempfile, logging
import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from transformers.utils import logging as hfl
hfl.set_verbosity_warning()
torch.manual_seed(0)
NL, TRUNC = 8, 6  # keep layers[:6] -> last kept layer index 5 (linear_attention, like L40)
text = dict(vocab_size=1000, hidden_size=128, intermediate_size=256, num_hidden_layers=NL, num_attention_heads=4,
            num_key_value_heads=2, head_dim=64, linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4, tie_word_embeddings=False,
            rope_parameters=dict(rope_type="default", rope_theta=1e7, partial_rotary_factor=0.25,
                                 mrope_interleaved=True, mrope_section=[3, 3, 2]),
            layer_types=(["linear_attention"] * 3 + ["full_attention"]) * (NL // 4))
cfg = Qwen3_5Config(text_config=text, vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                    out_hidden_size=128), tie_word_embeddings=False)
model = Qwen3_5ForConditionalGeneration(cfg).eval()
# make the GDN decay/gates non-trivial
with torch.no_grad():
    for n, p in model.named_parameters():
        if "norm" in n and p.dim() == 1: p.normal_(0, 0.1)
d = tempfile.mkdtemp(); model.save_pretrained(d)
print("attn impl:", model.config._attn_implementation)

def run(m, ids, layers, **kw):
    lm = m.model.language_model
    out = {}
    hs = [lm.layers[L].register_forward_hook(lambda mod, i, o, L=L: out.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach().clone()))
          for L in layers if L < len(lm.layers)]
    with torch.no_grad():
        r = m(input_ids=ids, use_cache=kw.pop("use_cache", False), logits_to_keep=1, **kw)
    for h in hs: h.remove()
    return out, r

def rel(a, b): return ((a.float() - b.float()).norm() / b.float().norm()).item()

ids = torch.randint(0, 1000, (1, 300))
ref, _ = run(model, ids, range(NL))

# --- 1. early exit by ModuleList slicing
m2 = copy.deepcopy(model); lm2 = m2.model.language_model
lm2.layers = lm2.layers[:TRUNC]
o, _ = run(m2, ids, range(TRUNC))
print("[slice] max|diff| layer", TRUNC - 1, (o[TRUNC - 1] - ref[TRUNC - 1]).abs().max().item())

# --- 2. early exit by loading truncated config (weights for later layers never instantiated)
c3 = Qwen3_5Config.from_pretrained(d)
c3.text_config.num_hidden_layers = TRUNC; c3.text_config.layer_types = c3.text_config.layer_types[:TRUNC]
m3, info = Qwen3_5ForConditionalGeneration.from_pretrained(d, config=c3, output_loading_info=True)
m3.eval()
print("[trunc-load] n_layers", len(m3.model.language_model.layers), "missing", sorted(info["missing_keys"])[:5],
      "n_unexpected", len(info["unexpected_keys"]), sorted(info["unexpected_keys"])[:2])
o, _ = run(m3, ids, range(TRUNC))
print("[trunc-load] max|diff| layer", TRUNC - 1, (o[TRUNC - 1] - ref[TRUNC - 1]).abs().max().item())
print("[trunc-load] lm_head/norm identical:", torch.equal(m3.lm_head.weight, model.lm_head.weight),
      torch.equal(m3.model.language_model.norm.weight, model.model.language_model.norm.weight))

# --- 3. prefix sharing
def prefix_test(m, dtype, P, sA, sB, tag):
    m = copy.deepcopy(m).to(dtype)
    pre = torch.randint(0, 1000, (1, P)); A = torch.randint(0, 1000, (1, sA)); B = torch.randint(0, 1000, (1, sB))
    res = {}
    for name, suf in (("A", A), ("B", B)):
        full, _ = run(m, torch.cat([pre, suf], 1), [TRUNC - 1, NL - 1])
        res[name] = {L: v[:, P:] for L, v in full.items()}
    _, r = run(m, pre, [], use_cache=True)
    cache = r.past_key_values
    out = {}
    for name, suf in (("A", A), ("B", B)):
        c = copy.deepcopy(cache)
        o, _ = run(m, suf, [TRUNC - 1, NL - 1], use_cache=True, past_key_values=c)
        out[name] = o
    for name in "AB":
        print(f"[prefix {tag} {dtype} P={P}] branch {name}: rel err L{TRUNC-1}={rel(out[name][TRUNC-1], res[name][TRUNC-1]):.2e}"
              f" L{NL-1}={rel(out[name][NL-1], res[name][NL-1]):.2e}")
    # also check that cache was not mutated (branch B after A gives the same as fresh)
    # batch both branches (equal length) via reorder_cache([0,0])
    if sA == sB:
        c = copy.deepcopy(cache); c.reorder_cache(torch.tensor([0, 0]))
        o, _ = run(m, torch.cat([A, B], 0), [TRUNC - 1], use_cache=True, past_key_values=c)
        print(f"[prefix-batched {tag} {dtype}] rel err A={rel(o[TRUNC-1][0:1], res['A'][TRUNC-1]):.2e} B={rel(o[TRUNC-1][1:2], res['B'][TRUNC-1]):.2e}")
    try:
        c = copy.deepcopy(cache); c.batch_repeat_interleave(2); print("batch_repeat_interleave ok")
    except Exception as e:
        print("batch_repeat_interleave FAILS:", type(e).__name__, e)
    try:
        _, r = run(m, pre, [], use_cache=True)
        o, _ = run(m, A, [TRUNC - 1], use_cache=True, past_key_values=r.past_key_values)
        o2, _ = run(m, B, [TRUNC - 1], use_cache=True, past_key_values=r.past_key_values)
        print(f"[reuse-without-copy] branch B rel err {rel(o2[TRUNC-1], res['B'][TRUNC-1]):.2e} (should be large: cache mutated in place)")
    except Exception as e:
        print("reuse-without-copy raised", type(e).__name__, e)

prefix_test(model, torch.float32, 200, 100, 77, "uneven")
prefix_test(model, torch.float32, 192, 64, 64, "aligned")
prefix_test(model, torch.bfloat16, 200, 100, 100, "uneven")
prefix_test(model, torch.bfloat16, 1000, 300, 300, "long")

# --- 4. padding
def pad_test(dtype):
    m = copy.deepcopy(model).to(dtype)
    a = torch.randint(1, 1000, (1, 300)); b = torch.randint(1, 1000, (1, 230))
    ra, _ = run(m, a, [TRUNC - 1]); rb, _ = run(m, b, [TRUNC - 1])
    n = 300 - 230
    # right pad
    ids = torch.cat([a, torch.cat([b, torch.zeros(1, n, dtype=torch.long)], 1)], 0)
    am = torch.ones_like(ids); am[1, 230:] = 0
    o, _ = run(m, ids, [TRUNC - 1], attention_mask=am)
    print(f"[right-pad {dtype}] rel err a={rel(o[TRUNC-1][0], ra[TRUNC-1][0]):.2e} b={rel(o[TRUNC-1][1, :230], rb[TRUNC-1][0]):.2e}")
    o, _ = run(m, ids, [TRUNC - 1])
    print(f"[right-pad no mask {dtype}] b={rel(o[TRUNC-1][1, :230], rb[TRUNC-1][0]):.2e}")
    # left pad
    ids = torch.cat([a, torch.cat([torch.zeros(1, n, dtype=torch.long), b], 1)], 0)
    am = torch.ones_like(ids); am[1, :n] = 0
    o, _ = run(m, ids, [TRUNC - 1], attention_mask=am)
    print(f"[left-pad {dtype}] rel err a={rel(o[TRUNC-1][0], ra[TRUNC-1][0]):.2e} b={rel(o[TRUNC-1][1, n:], rb[TRUNC-1][0]):.2e}")
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    o, _ = run(m, ids, [TRUNC - 1], attention_mask=am, position_ids=pos)
    print(f"[left-pad+pos_ids {dtype}] b={rel(o[TRUNC-1][1, n:], rb[TRUNC-1][0]):.2e}")
    # packing (position ids reset, no mask)
    ids = torch.cat([a, b], 1); pos = torch.cat([torch.arange(300), torch.arange(230)])[None]
    o, _ = run(m, ids, [TRUNC - 1], position_ids=pos)
    print(f"[packed via position_ids {dtype}] a={rel(o[TRUNC-1][0, :300], ra[TRUNC-1][0]):.2e} b={rel(o[TRUNC-1][0, 300:], rb[TRUNC-1][0]):.2e}")
pad_test(torch.float32); pad_test(torch.bfloat16)
