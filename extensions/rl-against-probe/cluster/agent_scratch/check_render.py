"""CPU only: harmony vs HF chat-template prompt ids for gpt-oss; special token ids."""
import json, sys, os
sys.path.insert(0, "/data/agastyas/rh_rl")
os.environ.setdefault("VLLM_SYSTEM_START_DATE", "2026-10-14")
from openai_harmony import Message, Role, load_harmony_encoding, HarmonyEncodingName
from vllm.entrypoints.openai.parser.harmony_utils import build_harmony_preamble, render_for_completion, get_encoding
from transformers import AutoTokenizer
from env_lcb import user_prompt
SNAP = "/data/agastyas/hf_cache_latekey/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
tok = AutoTokenizer.from_pretrained(SNAP)
enc = get_encoding()
t = json.loads(open("/data/agastyas/rh_rl/pilot_tasks.jsonl").readline())
up = user_prompt(t, "lenient")
for effort in ["low", "medium", "high"]:
    msgs = build_harmony_preamble(reasoning_effort=effort) + [Message.from_role_and_content(Role.USER, up)]
    h = render_for_completion(msgs)
    hf = tok.apply_chat_template([{"role": "user", "content": up}], tokenize=False, add_generation_prompt=True,
                                 reasoning_effort=effort)
    hf_ids = tok(hf, add_special_tokens=False)["input_ids"]
    same = h == hf_ids
    print(effort, "harmony_len", len(h), "hf_len", len(hf_ids), "identical", same)
    if not same:
        i = next((k for k in range(min(len(h), len(hf_ids))) if h[k] != hf_ids[k]), min(len(h), len(hf_ids)))
        print("  first diff at", i)
        print("  harmony:", repr(enc.decode(h[max(0, i-5):i+40])))
        print("  hf     :", repr(tok.decode(hf_ids[max(0, i-5):i+40])))
print("harmony head:", repr(enc.decode(h[:80])))
print("harmony tail:", repr(enc.decode(h[-12:])), h[-6:])
print("HF tokenizer == harmony on decode/encode of text:", tok(enc.decode(h), add_special_tokens=False)["input_ids"] == h)
for s in ["<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|return|>", "<|call|>", "<|constrain|>", "<|endoftext|>", "<|startoftext|>"]:
    print(s, tok.convert_tokens_to_ids(s), enc.encode(s, allowed_special="all"))
print("harmony stop tokens for assistant actions:", enc.stop_tokens_for_assistant_actions(), "stop tokens:", enc.stop_tokens())
print("'analysis' ids", enc.encode("analysis"), "'final' ids", enc.encode("final"), "'assistant'", enc.encode("assistant"))
