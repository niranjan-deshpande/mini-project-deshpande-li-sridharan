"""GPU TEST 2 (UNTESTED sketch; 1 GPU, ~10-15 min). gpt-oss-120b trainer with MXFP4-resident experts (dequantised per layer
per call), flex attention (sinks), attention-only LoRA r=32, non-reentrant grad checkpointing.
Reports: load time, dequant ms/layer, fwd+bwd time & peak mem at T in {4k,8k,16k,32k}, truncated (layers 0-13) probe-pass
time, flex-vs-eager LoRA-grad agreement at T=1024, all-layer grad coverage.
Run: CUDA_VISIBLE_DEVICES=<agreed gpu> HF_HUB_OFFLINE=1 /data/agastyas/rh_rl/venv/bin/python -I trainer_smoke.py"""
import json, sys, time, torch
sys.path.insert(0, "/data/agastyas/rh_rl/tests")
from dq_experts import make_dq
from safetensors import safe_open
from transformers import AutoConfig, GptOssForCausalLM
from dq_experts import convert_moe_packed_tensors
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
SNAP = "/data/agastyas/hf_cache_latekey/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
DEV = torch.device("cuda", 0); torch.cuda.set_device(DEV)
t0 = time.time()
cfg = AutoConfig.from_pretrained(SNAP)
if hasattr(cfg, "quantization_config"): del cfg.quantization_config
cfg.use_cache = False
with torch.device("meta"):
    model = GptOssForCausalLM._from_config(cfg, attn_implementation="flex_attention", experts_implementation="grouped_mm",
                                           dtype=torch.bfloat16)
for l in model.model.layers:
    del l.mlp.experts.gate_up_proj, l.mlp.experts.down_proj          # never allocate the 229 GB bf16 expert stack
model.to_empty(device=DEV)
model.model.rotary_emb = GptOssRotaryEmbedding(cfg, device=DEV)      # to_empty left inv_freq uninitialised
P = dict(model.named_parameters()); raw, loaded = {}, set()
wm = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
for f in sorted(set(wm.values())):
    with safe_open(f"{SNAP}/{f}", "pt", device="cuda:0") as fh:
        for k in fh.keys():
            if k.endswith(("_blocks", "_scales")): raw[k] = fh.get_tensor(k)
            else:
                with torch.no_grad(): P[k].copy_(fh.get_tensor(k))
                loaded.add(k)
assert set(P) == loaded, set(P) ^ loaded
for i, l in enumerate(model.model.layers):
    p = f"model.layers.{i}.mlp.experts."
    make_dq(l.mlp.experts, raw[p + "gate_up_proj_blocks"], raw[p + "gate_up_proj_scales"], raw[p + "down_proj_blocks"], raw[p + "down_proj_scales"])
del raw
print(f"load {time.time() - t0:.0f}s, resident {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

e0 = model.model.layers[0].mlp.experts
for _ in range(2): convert_moe_packed_tensors(e0.gu_blocks, e0.gu_scales); convert_moe_packed_tensors(e0.d_blocks, e0.d_scales)
torch.cuda.synchronize(); t = time.time()
for _ in range(5): convert_moe_packed_tensors(e0.gu_blocks, e0.gu_scales); convert_moe_packed_tensors(e0.d_blocks, e0.d_scales)
torch.cuda.synchronize(); print(f"dequant per layer {(time.time() - t) / 5 * 1e3:.1f} ms", flush=True)

model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
for n, p in model.named_parameters():
    if "lora_B" in n: torch.nn.init.normal_(p, std=1e-3)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
inner = model.base_model.model; lm, head = inner.model, inner.lm_head
PL = 13

class Stop(Exception): pass
def probe_pass(x):                    # layers 0..13 only
    cap = {}
    def hk(m, i, o): cap["h"] = o[0] if isinstance(o, tuple) else o; raise Stop
    h = lm.layers[PL].register_forward_hook(hk)
    try: lm(input_ids=x)
    except Stop: pass
    finally: h.remove()
    return cap["h"]

def lp_loss(x):
    hs = lm(input_ids=x).last_hidden_state[0, :-1]; tgt = x[0, 1:]; tot = 0
    for s in range(0, hs.shape[0], 4096):
        def f(hc, tc):
            lg = head(hc).float(); return (lg.gather(-1, tc[:, None])[:, 0] - torch.logsumexp(lg, -1)).sum()
        tot = tot + checkpoint(f, hs[s:s + 4096], tgt[s:s + 4096], use_reentrant=False)
    return -tot / tgt.numel()

g = torch.Generator(device=DEV).manual_seed(0)
def rep(tag):
    def h(m, i, o): print(f"{tag} alloc {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    return h
hs_ = [lm.layers[k].register_forward_hook(rep(f"fwd L{k}")) for k in (0, 1, 2, 5, 10, 20, 35)]
model.train()
x = torch.randint(0, 200000, (1, 2048), device=DEV, generator=g)
print("training", lm.layers[0].training, "gc", getattr(lm.layers[0], "gradient_checkpointing", None), getattr(lm, "gradient_checkpointing", None), flush=True)
loss = lp_loss(x)
print(f"after fwd alloc {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
for h in hs_: h.remove()
