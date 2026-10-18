# CPU-only: tiny GptOss, PEFT attention + 3D-expert LoRA, adapter key format, grad flow, MXFP4 on-the-fly-dequant autograd check.
import os, json, torch, safetensors.torch as st
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
from transformers import GptOssConfig, GptOssForCausalLM
from peft import LoraConfig, get_peft_model
torch.manual_seed(0)
cfg = GptOssConfig(num_hidden_layers=2, hidden_size=64, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
                   head_dim=16, num_local_experts=8, num_experts_per_tok=2, vocab_size=128, sliding_window=8,
                   layer_types=["sliding_attention", "full_attention"])
m = GptOssForCausalLM(cfg).float()
print("experts impl:", m.config._experts_implementation, "attn:", m.config._attn_implementation)
names = sorted({n.split(".")[-1] for n, _ in m.named_modules()}); print("module leaf names:", names)
print("expert params:", [(n, tuple(p.shape)) for n, p in m.model.layers[0].mlp.named_parameters()])
for tgt, tp in [("attn", None), ("attn+experts", ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"])]:
    mm = GptOssForCausalLM(cfg).float(); mm.load_state_dict(m.state_dict())
    lc = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], target_parameters=tp)
    pm = get_peft_model(mm, lc)
    for n, p in pm.named_parameters():
        if "lora_B" in n: torch.nn.init.normal_(p, std=0.02)
    ntrain = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    x = torch.randint(0, 128, (1, 24))
    out = pm(input_ids=x).logits
    out.float().logsumexp(-1).sum().backward()
    g = {n: float(p.grad.norm()) for n, p in pm.named_parameters() if p.requires_grad and p.grad is not None}
    print(f"[{tgt}] trainable={ntrain} params_with_grad={len(g)} zero_grad={[n for n,v in g.items() if v==0][:4]}")
    d = f"/data/agastyas/rh_rl/agent_scratch/tiny_adapter_{tgt.replace(chr(43), chr(95))}"
    pm.save_pretrained(d)
    sd = st.load_file(d + "/adapter_model.safetensors")
    print(" adapter keys (layer0):", sorted((k, tuple(v.shape)) for k, v in sd.items() if ".layers.0." in k))
    print(" adapter_config target_modules/params:", {k: v for k, v in json.load(open(d + "/adapter_config.json")).items() if k.startswith("target")})

# MXFP4-resident experts: on-the-fly dequant inside an autograd.Function, compare grads wrt input against bf16 reference
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
E, I, H = 8, 64, 64
blocks_gu = torch.randint(0, 256, (E, 2 * I, H // 32, 16), dtype=torch.uint8); scales_gu = torch.randint(120, 130, (E, 2 * I, H // 32), dtype=torch.uint8)
W = convert_moe_packed_tensors(blocks_gu, scales_gu, dtype=torch.float32)   # (E, H, 2I) as GptOssExperts expects
print("dequant shape", tuple(W.shape), "matches GptOssExperts.gate_up_proj", (E, H, 2 * I))
class DQMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, e, blocks, scales):
        w = convert_moe_packed_tensors(blocks, scales, dtype=x.dtype)[e]; ctx.save_for_backward(e, blocks, scales); return x @ w
    @staticmethod
    def backward(ctx, gy):
        e, blocks, scales = ctx.saved_tensors
        w = convert_moe_packed_tensors(blocks, scales, dtype=gy.dtype)[e]      # recompute, never stored across layers
        return gy @ w.transpose(-1, -2), None, None, None
x = torch.randn(5, H, requires_grad=True); e = torch.tensor(3)
y = DQMatmul.apply(x, e, blocks_gu, scales_gu); y.square().sum().backward(); g1 = x.grad.clone(); x.grad = None
(x @ W[3]).square().sum().backward()
print("custom-backward dX max err vs reference:", float((g1 - x.grad).abs().max()))
