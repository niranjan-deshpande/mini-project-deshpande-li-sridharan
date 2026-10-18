import os, glob, json, tempfile, torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, AutoModelForCausalLM, AutoConfig
from transformers.utils import logging as hfl
hfl.set_verbosity_error()
torch.manual_seed(0)
NL, TRUNC = 8, 6
text = dict(vocab_size=1000, hidden_size=128, intermediate_size=256, num_hidden_layers=NL, num_attention_heads=4,
            num_key_value_heads=2, head_dim=64, linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=32, linear_value_head_dim=32, tie_word_embeddings=False,
            rope_parameters=dict(rope_type="default", rope_theta=1e7, partial_rotary_factor=0.25, mrope_interleaved=True, mrope_section=[3, 3, 2]),
            layer_types=(["linear_attention"] * 3 + ["full_attention"]) * (NL // 4))
cfg = Qwen3_5Config(text_config=text, vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=128), tie_word_embeddings=False)
model = Qwen3_5ForConditionalGeneration(cfg).eval()
d = tempfile.mkdtemp(); model.save_pretrained(d, max_shard_size="150KB")
idx = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
ids = torch.randint(0, 1000, (1, 100))
ref = {}
h = model.model.language_model.layers[TRUNC-1].register_forward_hook(lambda m, i, o: ref.__setitem__("x", o))
with torch.no_grad(): model(ids, use_cache=False)
h.remove()

# AutoModelForCausalLM path (what t2_acts.py does)
m = AutoModelForCausalLM.from_pretrained(d)
print("AutoModelForCausalLM ->", type(m).__name__, "| has visual:", any("visual" in n for n, _ in m.named_parameters()),
      "| lm_head equal:", torch.equal(m.lm_head.weight, model.lm_head.weight))
# truncated config via AutoConfig + text_config
c = AutoConfig.from_pretrained(d)
c.text_config.num_hidden_layers = TRUNC; c.text_config.layer_types = c.text_config.layer_types[:TRUNC]
m = AutoModelForCausalLM.from_pretrained(d, config=c)
print("trunc via AutoConfig: type", type(m).__name__, "n layers", len(m.model.layers), "cfg nl", m.config.num_hidden_layers)
out = {}
m.model.layers[TRUNC-1].register_forward_hook(lambda mm, i, o: out.__setitem__("x", o))
with torch.no_grad(): m(ids, use_cache=False, logits_to_keep=1)
print("trunc via AutoConfig: max|diff|", (out["x"] - ref["x"]).abs().max().item())

# delete shards that only hold layers >= TRUNC (and mtp/visual-only shards are kept)
needed = {v for k, v in idx.items() if not any(f"layers.{L}." in k and "language_model" in k for L in range(TRUNC, NL))}
allsh = set(idx.values()); drop = allsh - needed
print("shards", len(allsh), "dropping", len(drop))
for s in drop: os.remove(f"{d}/{s}")
try:
    m = AutoModelForCausalLM.from_pretrained(d, config=c)
    out = {}
    m.model.layers[TRUNC-1].register_forward_hook(lambda mm, i, o: out.__setitem__("x", o))
    with torch.no_grad(): m(ids, use_cache=False)
    print("load with later-layer shards deleted: OK, max|diff|", (out["x"] - ref["x"]).abs().max().item())
except Exception as e:
    print("load with later-layer shards deleted FAILS:", type(e).__name__, str(e)[:300])
# rewrite index to only list existing shards
j = json.load(open(f"{d}/model.safetensors.index.json"))
j["weight_map"] = {k: v for k, v in j["weight_map"].items() if os.path.exists(f"{d}/{v}")}
json.dump(j, open(f"{d}/model.safetensors.index.json", "w"))
m = AutoModelForCausalLM.from_pretrained(d, config=c)
out = {}
m.model.layers[TRUNC-1].register_forward_hook(lambda mm, i, o: out.__setitem__("x", o))
with torch.no_grad(): m(ids, use_cache=False)
print("load with pruned index: OK, max|diff|", (out["x"] - ref["x"]).abs().max().item(), "lm_head eq", torch.equal(m.lm_head.weight, model.lm_head.weight), "norm eq", torch.equal(m.model.norm.weight, model.model.language_model.norm.weight))
