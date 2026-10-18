"""CPU only: what PEFT 0.21 + transformers 5.18 save for gpt-oss LoRA (attention modules + 3D expert params)."""
import torch, json, inspect
from transformers import GptOssConfig, GptOssForCausalLM
from transformers.models.gpt_oss import modeling_gpt_oss as M
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
cfg = GptOssConfig(num_hidden_layers=2, hidden_size=64, intermediate_size=48, num_local_experts=4, num_experts_per_tok=2,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=256, layer_types=["sliding_attention", "full_attention"])
m = GptOssForCausalLM(cfg)
print("expert params:", [(n, tuple(p.shape)) for n, p in m.model.layers[0].mlp.experts.named_parameters()])
print("router:", type(m.model.layers[0].mlp.router).__name__, [(n, tuple(p.shape)) for n, p in m.model.layers[0].mlp.router.named_parameters()])
print("attn modules:", [n for n, _ in m.model.layers[0].self_attn.named_children()])
src = inspect.getsource(M.GptOssExperts.forward)
print("experts.forward uses:", [l.strip() for l in src.splitlines() if "gate_up_proj" in l or "down_proj" in l][:6])
r = 4
for name, kw in [("attn_only", dict(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])),
                 ("attn+experts", dict(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                       target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]))]:
    pm = get_peft_model(GptOssForCausalLM(cfg), LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, **kw))
    sd = get_peft_model_state_dict(pm)
    print("==", name, "n_keys", len(sd))
    for k, v in sd.items():
        if ".layers.0." in k:
            print("  ", k, tuple(v.shape))
