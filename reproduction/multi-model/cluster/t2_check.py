"""Tokenizer-only check that t2_acts regions match the exported passage texts."""
import json, sys, re, importlib.util, types
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
src = open("/data/agastyas/rh_probe/t2_acts.py").read()
ns = {}
exec(src[src.index("def to_chat"):src.index("rows = [json.loads")], ns)
norm = lambda s: re.sub(r"\s+", " ", s).strip()
bad = tot = 0
for l in list(open(sys.argv[2]))[:int(sys.argv[3])]:
    r = json.loads(l)
    text = tok.apply_chat_template(ns["to_chat"](r), tools=r.get("tools") or None, tokenize=False, add_generation_prompt=False, preserve_thinking=True)
    reg = ns["regions"](text, r["n_steps"])
    for p in r["passages"]:
        tot += 1
        g = reg.get((p["step"], p["channel"]))
        got = norm(text[g[0]:g[1]]) if g else None
        want = norm(p["text"])
        ok = got == want if p["channel"] != "tool_calls" else (got is not None and "<tool_call>" in got)
        if not ok:
            bad += 1
            if bad <= 4: print("MISMATCH", r["uid"], p["n"], p["channel"], "\n got:", (got or "")[:200], "\n want:", want[:200])
    print(r["uid"], "tokens", len(tok(text)["input_ids"]))
print(f"{bad}/{tot} mismatched")
