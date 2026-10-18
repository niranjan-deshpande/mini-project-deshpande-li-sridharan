"""CPU: random attention-only PEFT LoRA adapter for gpt-oss-120b (for GPU test 1/3). B ~ N(0, std) so it perturbs outputs."""
import json, os, sys, torch, safetensors.torch as st
r, std, out = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
assert out.startswith("/data/agastyas/rh_rl/agent_scratch/")
H, NQ, NKV, D, L = 2880, 64, 8, 64, 36
shapes = {"q_proj": (H, NQ * D), "k_proj": (H, NKV * D), "v_proj": (H, NKV * D), "o_proj": (NQ * D, H)}
g = torch.Generator().manual_seed(0); sd = {}
for i in range(L):
    for m, (fin, fout) in shapes.items():
        p = f"base_model.model.model.layers.{i}.self_attn.{m}"
        sd[p + ".lora_A.weight"] = (torch.randn(r, fin, generator=g) / fin ** 0.5).to(torch.bfloat16)
        sd[p + ".lora_B.weight"] = (torch.randn(fout, r, generator=g) * std).to(torch.bfloat16)
os.makedirs(out, exist_ok=True)
st.save_file(sd, f"{out}/adapter_model.safetensors")
json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": 2 * r, "lora_dropout": 0.0, "bias": "none",
           "target_modules": list(shapes), "base_model_name_or_path": "openai/gpt-oss-120b", "use_rslora": False,
           "use_dora": False, "fan_in_fan_out": False, "modules_to_save": None}, open(f"{out}/adapter_config.json", "w"), indent=1)
print(len(sd), "tensors,", sum(v.numel() for v in sd.values()) / 1e6, "M params")
