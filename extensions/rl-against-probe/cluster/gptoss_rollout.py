"""Exact-token gpt-oss rollouts from a vLLM server (/v1/completions with a harmony-rendered token-id prompt).

prompt_ids(text, effort)  -> harmony ids ending in <|start|>assistant (identical to HF chat template / vLLM chat path)
sample(port, ids, n, ...) -> [{"c_ids", "vlp", "finish", "analysis", "final"}]; c_ids includes the closing <|return|>
"""
import json, urllib.request
from openai_harmony import (Conversation, HarmonyEncodingName, Message, ReasoningEffort, Role, StreamableParser,
                            SystemContent, load_harmony_encoding)

ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
RETURN, CALL = 200002, 200012                      # <|return|>, <|call|>  (== ENC.stop_tokens_for_assistant_actions())
EFFORT = {"low": ReasoningEffort.LOW, "medium": ReasoningEffort.MEDIUM, "high": ReasoningEffort.HIGH}
START_DATE = "2026-10-14"                           # pin it: vLLM/HF otherwise insert today's date into the system msg


def prompt_ids(user_text, effort="medium", date=START_DATE):
    sys = SystemContent.new().with_reasoning_effort(EFFORT[effort]).with_conversation_start_date(date)
    conv = Conversation.from_messages([Message.from_role_and_content(Role.SYSTEM, sys),
                                       Message.from_role_and_content(Role.USER, user_text)])
    return ENC.render_conversation_for_completion(conv, Role.ASSISTANT)


def split_channels(c_ids):
    """(analysis_text, final_text|None). Token-level; tolerant of truncation (final=None if no final channel)."""
    p = StreamableParser(ENC, role=Role.ASSISTANT)
    for t in c_ids:
        p.process(t)
    ana, fin, has_final = [], [], False
    for m in p.messages:
        txt = "".join(c.text for c in m.content)
        if m.channel == "final":
            fin.append(txt); has_final = True
        else:
            ana.append(txt)
    if p.current_channel == "final":
        fin.append(p.current_content); has_final = True
    elif p.current_content:
        ana.append(p.current_content)
    return "".join(ana), ("".join(fin) if has_final else None)


def sample(port, ids, n, key, model="gpt-oss-120b", max_tokens=12288, timeout=7200):
    body = {"model": model, "prompt": ids, "n": n, "max_tokens": max_tokens, "temperature": 1.0, "top_p": 1.0,
            "top_k": -1, "logprobs": 0, "return_token_ids": True, "skip_special_tokens": False,
            "stop_token_ids": [RETURN, CALL]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    res = json.load(urllib.request.urlopen(req, timeout=timeout))
    out = []
    for c in res["choices"]:
        assert c["prompt_token_ids"] == ids
        c_ids, vlp = c["token_ids"], c["logprobs"]["token_logprobs"]
        assert len(c_ids) == len(vlp)
        ana, fin = split_channels(c_ids)
        ok = c["finish_reason"] == "stop" and c_ids[-1] == RETURN      # treat truncated / tool-call as no answer
        out.append({"c_ids": c_ids, "vlp": vlp, "finish": c["finish_reason"], "analysis": ana,
                    "final": fin if ok else None})
    return out


if __name__ == "__main__":
    import sys
    key = open("/data/agastyas/rh_probe/api_key").read().strip()
    from vllm.entrypoints.openai.parser.harmony_utils import build_harmony_preamble, render_for_completion
    import os
    os.environ["VLLM_SYSTEM_START_DATE"] = START_DATE
    q = "What is 17*3? Reply with just the number."
    ref = render_for_completion(build_harmony_preamble(reasoning_effort="low") + [Message.from_role_and_content(Role.USER, q)])
    ids = prompt_ids(q, "low")
    print("matches vLLM chat-path rendering:", ids == ref)
    if len(sys.argv) > 1:
        for s in sample(sys.argv[1], ids, 2, key, max_tokens=64):
            print(s["finish"], len(s["c_ids"]), s["c_ids"][-1], round(sum(s["vlp"]), 3), repr(s["analysis"][:60]), repr(s["final"]))
