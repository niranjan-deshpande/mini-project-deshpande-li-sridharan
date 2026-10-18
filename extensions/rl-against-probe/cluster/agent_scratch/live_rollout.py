"""Tiny live check of the exact-token rollout path against the running gpt-oss-120b server (no LoRA).
CPU only on this side; sends ONE /v1/completions request (n=2, max_tokens=64)."""
import json, os, sys, time, urllib.request
os.environ.setdefault("VLLM_SYSTEM_START_DATE", "2026-10-14")
from openai_harmony import Message, Role
from vllm.entrypoints.openai.parser.harmony_utils import build_harmony_preamble, render_for_completion, get_encoding

RETURN, CALL, END = 200002, 200012, 200007
enc = get_encoding()
KEY = open("/data/agastyas/rh_probe/api_key").read().strip()


def prompt_ids(user_text, effort):
    msgs = build_harmony_preamble(reasoning_effort=effort) + [Message.from_role_and_content(Role.USER, user_text)]
    return render_for_completion(msgs)          # ends with <|start|>assistant


def split_channels(c_ids):
    """(analysis_text, final_text or None, parsed_ok). Token-level parse with openai_harmony; the final answer
    exists only if a final-channel message is present (truncated samples -> None)."""
    try:
        msgs = enc.parse_messages_from_completion_tokens(c_ids, Role.ASSISTANT, strict=False)
    except Exception as e:
        return None, None, f"parse_error: {e}"
    ana = "".join(c.text for m in msgs if m.channel == "analysis" for c in m.content)
    fin = [c.text for m in msgs if m.channel == "final" for c in m.content]
    return ana, ("".join(fin) if fin else None), "ok"


p = prompt_ids("What is 17*3? Reply with just the number.", "low")
body = {"model": "gpt-oss-120b", "prompt": p, "n": 2, "max_tokens": 64, "temperature": 1.0, "top_p": 1.0,
        "logprobs": 0, "return_token_ids": True, "skip_special_tokens": False,
        "stop_token_ids": [RETURN, CALL]}
req = urllib.request.Request("http://127.0.0.1:8400/v1/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
t0 = time.time()
res = json.load(urllib.request.urlopen(req, timeout=300))
print("latency_s", round(time.time() - t0, 2), "usage", res.get("usage"))
for c in res["choices"]:
    ids, lp = c["token_ids"], c["logprobs"]["token_logprobs"]
    print("---- choice", c["index"], "finish", c["finish_reason"], "stop_reason", c.get("stop_reason"))
    print("prompt_token_ids echoed & equal:", c.get("prompt_token_ids") == p if c.get("prompt_token_ids") is not None else "not in choice")
    print("n_tokens", len(ids), "n_logprobs", len(lp), "any None", any(x is None for x in lp))
    print("first ids", ids[:6], "last ids", ids[-4:])
    print("text repr:", repr(c["text"][:300]))
    print("decode(ids)==text:", enc.decode(ids) == c["text"], "| tokens field sample:", c["logprobs"]["tokens"][:4])
    print("split:", split_channels(ids))
print("top-level prompt_token_ids?", "prompt_token_ids" in res)
