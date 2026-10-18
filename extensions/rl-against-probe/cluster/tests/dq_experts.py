"""Sketch: keep gpt-oss experts in MXFP4 (uint8 blocks + e8m0 scales, ~61 GB for 120b) and dequantise per layer per call.
Under layer-level gradient checkpointing the bf16 copy (~6.4 GB/layer for 120b) lives only while that layer runs/back-props."""
import torch
from transformers.integrations.mxfp4 import _convert_moe_packed_tensors
ROWS = 4 * 1024 * 1024  # dequant chunk (rows); the HF default (32M) needs ~10+ GB of int64 temporaries

def convert_moe_packed_tensors(b, s, dtype=torch.bfloat16):
    return _convert_moe_packed_tensors(b, s, dtype=dtype, rows_per_chunk=ROWS)  # no silent CPU fallback

def make_dq(experts, gu_blocks, gu_scales, d_blocks, d_scales):
    """experts: a transformers GptOssExperts (decorated with use_experts_implementation). Replaces its two 3-D weight
    Parameters with MXFP4 buffers; forward pre-hook materialises bf16 weights, post-hook drops them."""
    for n in ("gate_up_proj", "down_proj"):
        if n in experts._parameters or hasattr(experts, n):
            delattr(experts, n)
    for n, t in dict(gu_blocks=gu_blocks, gu_scales=gu_scales, d_blocks=d_blocks, d_scales=d_scales).items():
        experts.register_buffer(n, t, persistent=False)
    def pre(mod, args):
        dt = args[0].dtype
        with torch.no_grad():
            mod.gate_up_proj = convert_moe_packed_tensors(mod.gu_blocks, mod.gu_scales, dtype=dt)  # (E, H, 2I)
            mod.down_proj = convert_moe_packed_tensors(mod.d_blocks, mod.d_scales, dtype=dt)       # (E, I, H)
    def post(mod, args, out):
        del mod.gate_up_proj, mod.down_proj
    experts.register_forward_pre_hook(pre); experts.register_forward_hook(post)
    return experts

if __name__ == "__main__":
    import os, copy
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    from transformers import GptOssConfig, GptOssForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    cfg = GptOssConfig(num_hidden_layers=2, hidden_size=64, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
                       head_dim=16, num_local_experts=8, num_experts_per_tok=2, vocab_size=128, sliding_window=8,
                       layer_types=["sliding_attention", "full_attention"])
    ref = GptOssForCausalLM(cfg).float(); dq = copy.deepcopy(ref)
    E, H, I = 8, 64, 64
    for lr, ld in zip(ref.model.layers, dq.model.layers):
        gb = torch.randint(0, 256, (E, 2 * I, H // 32, 16), dtype=torch.uint8); gs = torch.randint(122, 128, (E, 2 * I, H // 32), dtype=torch.uint8)
        db = torch.randint(0, 256, (E, H, I // 32, 16), dtype=torch.uint8); ds = torch.randint(122, 128, (E, H, I // 32), dtype=torch.uint8)
        with torch.no_grad():
            lr.mlp.experts.gate_up_proj.copy_(convert_moe_packed_tensors(gb, gs, dtype=torch.float32))
            lr.mlp.experts.down_proj.copy_(convert_moe_packed_tensors(db, ds, dtype=torch.float32))
        make_dq(ld.mlp.experts, gb, gs, db, ds)
    outs = []
    for m in (ref, dq):
        torch.manual_seed(1)
        pm = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        for n, p in pm.named_parameters():
            if "lora_B" in n: torch.nn.init.normal_(p, std=0.05)
        pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); pm.enable_input_require_grads(); pm.train()
        x = torch.randint(0, 128, (1, 32), generator=torch.Generator().manual_seed(2))
        lg = pm(input_ids=x).logits; lg.float().log_softmax(-1)[0, :-1].gather(-1, x[0, 1:, None]).sum().backward()
        outs.append((lg.detach(), {n: p.grad.clone() for n, p in pm.named_parameters() if p.requires_grad}))
    print("logit max diff:", float((outs[0][0] - outs[1][0]).abs().max()))
    print("lora grad max rel diff:", max(float((outs[0][1][n] - outs[1][1][n]).norm() / (outs[0][1][n].norm() + 1e-12)) for n in outs[0][1]))
    print("n lora grads:", len(outs[0][1]), "all nonzero:", all(float(g.norm()) > 0 for g in outs[1][1].values()))
