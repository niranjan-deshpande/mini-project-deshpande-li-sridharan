"""Export Inspect .eval logs to JSONL: structured chat messages (for activation extraction) plus the transcript
renderings used by the judge (paper Fig 20 format, with PASSAGE INDEX) and the generic monitor (Fig 17 format,
with passage_id headers).

Passages: one per (step, channel) for channel in reasoning / content / tool_calls (non-empty only). The monitor sees
individual tool calls as separate candidate passages; its flags are mapped back to the step's tool_calls passage.

Usage (athena01): venv/bin/python export_rollouts.py logs/qwen_pilot out/qwen_pilot.jsonl
"""
import json, sys
from pathlib import Path
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser

R_HDR = "--- REASONING (model-internal; not visible in the environment) ---"


def text_and_reasoning(msg):
    if isinstance(msg.content, str):
        return msg.content, ""
    text, reasoning = [], []
    for c in msg.content:
        if c.type == "reasoning":
            reasoning.append(c.reasoning or "")
        elif c.type == "text":
            text.append(c.text)
    return "\n".join(text).strip(), "\n".join(reasoning).strip()


def fmt_call(tc):
    args = tc.arguments or {}
    if tc.function == "bash" and "cmd" in args:
        return f"[bash]\n{args['cmd']}"
    if tc.function == "python" and "code" in args:
        return f"[python]\n{args['code']}"
    return f"[{tc.function}] {json.dumps(args, ensure_ascii=False)}"


def convert(sample, split, task):
    msgs = sample.messages
    chat, steps, system, task_text = [], [], "", ""
    feedback_seen = 0
    for i, m in enumerate(msgs):
        if isinstance(m, ChatMessageSystem):
            system = m.text
            chat.append({"role": "system", "content": m.text})
        elif isinstance(m, ChatMessageUser):
            if not task_text and not steps:
                task_text = m.text
            else:
                steps.append({"feedback": m.text})
                feedback_seen += 1
            chat.append({"role": "user", "content": m.text})
        elif isinstance(m, ChatMessageAssistant):
            text, reasoning = text_and_reasoning(m)
            calls = [{"id": tc.id, "name": tc.function, "arguments": tc.arguments or {}, "rendered": fmt_call(tc)}
                     for tc in (m.tool_calls or [])]
            steps.append({"reasoning": reasoning, "content": text, "calls": calls, "obs": {}})
            chat.append({"role": "assistant", "content": text, "reasoning": reasoning,
                         "tool_calls": [{"id": c["id"], "name": c["name"], "arguments": c["arguments"]} for c in calls]})
        elif isinstance(m, ChatMessageTool):
            for st in reversed(steps):
                if "calls" in st:
                    st["obs"][m.tool_call_id] = m.text if m.error is None else f"{m.text}\n[error] {m.error.message}"
                    break
            chat.append({"role": "tool", "tool_call_id": m.tool_call_id, "name": m.function, "content": m.text})

    # --- renderings ---
    score = next(iter(sample.scores.values())).value if sample.scores else None
    header = (f"=== TRANSCRIPT HEADER ===\ntask: {task}\nsample: {sample.id} epoch {sample.epoch}\n"
              f"=== SYSTEM PROMPT ===\n{system}\n=== TASK (user) ===\n{task_text}\n")
    jparts, mparts, passages = [header], [header], []
    n_step = 0
    for st in steps:
        if "feedback" in st:
            jparts.append(f"=== FEEDBACK ===\n{st['feedback']}\n")
            mparts.append(f"=== FEEDBACK ===\n{st['feedback']}\n")
            continue
        n_step += 1
        jparts.append(f"=== STEP {n_step} ===")
        mparts.append(f"=== STEP {n_step} ===")
        if st["reasoning"]:
            pid = f"S{n_step}R"
            passages.append({"n": len(passages) + 1, "step": n_step, "channel": "reasoning", "pid": [pid], "text": st["reasoning"]})
            jparts.append(f"{R_HDR}\n{st['reasoning']}")
            mparts.append(f"--- REASONING (model-internal; not visible in the environment) [passage_id={pid}] ---\n{st['reasoning']}")
        if st["content"]:
            pid = f"S{n_step}A"
            passages.append({"n": len(passages) + 1, "step": n_step, "channel": "content", "pid": [pid], "text": st["content"]})
            jparts.append(f"--- ASSISTANT ---\n{st['content']}")
            mparts.append(f"--- ASSISTANT [passage_id={pid}] ---\n{st['content']}")
        if st["calls"]:
            pids = []
            for k, c in enumerate(st["calls"], 1):
                pid = f"S{n_step}T{k}"
                pids.append(pid)
                obs = st["obs"].get(c["id"], "")
                jparts.append(f"--- COMMAND {k} ---\n{c['rendered']}\n--- OBSERVATION {k} ---\n{obs}")
                mparts.append(f"--- TOOL CALL {k} [passage_id={pid}] ---\n{c['rendered']}\n--- TOOL RESPONSE {k} ---\n{obs}")
            passages.append({"n": len(passages) + 1, "step": n_step, "channel": "tool_calls", "pid": pids,
                             "text": "\n".join(c["rendered"] for c in st["calls"])})
    exit_line = f"=== EXIT (score={score}, stop={getattr(sample, 'limit', None) and sample.limit.type}) ==="
    jparts.append(exit_line)
    mparts.append(exit_line)
    index = [f"=== PASSAGE INDEX ({len(passages)} passages) ==="] + \
            [f"P{p['n']} | step {p['step']} | {p['channel']} | {len(p['text'])}" for p in passages]
    return {
        "uid": f"{task}|{sample.id}|{sample.epoch}", "task": task, "split": split, "sample_id": sample.id,
        "epoch": sample.epoch, "score": score, "error": str(sample.error.message)[:300] if sample.error else None,
        "limit": sample.limit.type if getattr(sample, "limit", None) else None,
        "n_steps": n_step, "chat": chat, "passages": passages,
        "judge_text": "\n\n".join(jparts + index), "monitor_text": "\n\n".join(mparts),
        "metadata": {k: v for k, v in (sample.metadata or {}).items() if isinstance(v, (str, int, float, bool))},
    }


def main(log_dir, out):
    rows = []
    for f in sorted(Path(log_dir).glob("*.eval")):
        log = read_eval_log(str(f))
        split = log.eval.task_args.get("split") if log.eval.task_args else None
        for s in log.samples or []:
            rows.append(convert(s, split, log.eval.task))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"exported {len(rows)} samples -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
