"""Render SWE pairs with the Qwen chat template exactly as build_dom.py does, and locate turn/span tokens.

Shared by the CPU pre-check and the GPU harvest so both use identical positions.
"""
import re

THINK = re.compile(r"<think>(.*?)</think>\s*", re.S)
ASST_OPEN, TURN_END = "<|im_start|>assistant", "<|im_end|>"


def to_chat(messages):
    """Copied from reproduction/multi-model/cluster/build_dom.py (qwen branch)."""
    out = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "assistant":
            reasoning = "\n".join(t.strip() for t in THINK.findall(content))
            content = THINK.sub("", content).strip()
            msg = {"role": "assistant", "content": content}
            if reasoning:
                msg["reasoning_content"] = reasoning
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "content": content})
        else:
            out.append({"role": role, "content": content})
    return out


def locate(tok, branch):
    """(input_ids, {'turn': [...], 'span': [...]}) for a branch dict with messages, span, turn_index; None on failure."""
    msgs, k = branch["messages"], branch["turn_index"]
    text = tok.apply_chat_template(to_chat(msgs), tokenize=False, add_generation_prompt=False, preserve_thinking=True)
    n_asst = sum(m["role"] == "assistant" for m in msgs[: k + 1])
    starts = [m.start() for m in re.finditer(re.escape(ASST_OPEN), text)]
    if len(starts) < n_asst:
        return None
    t0 = starts[n_asst - 1]
    t1 = text.find(TURN_END, t0)
    t1 = len(text) if t1 < 0 else t1
    s0 = text.find(branch["span"], t0)
    if s0 < 0 or s0 > t1:
        return None
    s1 = s0 + len(branch["span"])
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    off = enc["offset_mapping"]
    span = [i for i, (x, y) in enumerate(off) if y > s0 and x < s1]
    turn = [i for i, (x, y) in enumerate(off) if y > t0 and x < t1]
    if not span or not turn:
        return None
    return enc["input_ids"], {"turn": turn, "span": span, "last": [span[-1]]}
