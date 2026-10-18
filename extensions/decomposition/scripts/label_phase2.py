"""Phase 2: label every passage of Qwen rollouts for contradiction_failure and eval_awareness (prompts/rubric_phase2.md).

One call per rollout per judge. Quotes are verified against the passage text (whitespace-normalised); unverified
flags are withdrawn. A label is positive iff at least one verified flag for it remains.
Usage: python scripts/label_phase2.py --run pilot --judge luna --uids data/phase2/pilot_uids.txt
Judges: luna (gpt-6-luna, OpenAI Responses API), haiku (anthropic/claude-haiku-5.5 via OpenRouter),
        dsflash (deepseek/deepseek-v4.1-flash via OpenRouter).
"""
import argparse, concurrent.futures as cf, gzip, json, os, re, sys, time, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLL = ROOT.parents[1] / "reproduction/multi-model/results/rollouts/qwen_rollouts.jsonl.gz"
RUBRIC = (ROOT / "prompts/rubric_phase2.md").read_text()
SCHEMA = json.loads((ROOT / "prompts/schema_phase2.json").read_text())
JUDGES = {"luna": ("openai", "gpt-6-luna", (0.10, 0.50)),
          "haiku": ("openrouter", "anthropic/claude-haiku-5.5", (0.10, 0.50)),
          "dsflash": ("openrouter", "deepseek/deepseek-v4.1-flash", (0.044, 0.765))}
LABELS = ("contradiction_failure", "eval_awareness")
norm = lambda s: re.sub(r"\s+", " ", s or "").strip()


def quote_ok(quote, text):
    """Verbatim (whitespace-normalised) match; a quote with '...' elisions passes if every fragment matches in order."""
    q, t = norm(quote), norm(text)
    if not q:
        return False
    frags = [f.strip() for f in re.split(r"\s*(?:\.\.\.|…)\s*", q) if f.strip()]
    pos = 0
    for f in frags:
        i = t.find(f, pos)
        if i < 0:
            return False
        pos = i + len(f)
    return True


def load_env():
    for line in open(ROOT.parents[1] / ".env"):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.strip().split("=", 1)
            if v:
                os.environ.setdefault(k, v.strip().strip('"'))


def tag_passages(text, passages):
    """Write each passage's id into its block header, e.g. '--- REASONING [P5] (...) ---', so the judge
    doesn't have to count blocks. Returns (text, number of tags inserted)."""
    pid = {(p["step"], p["channel"]): p["n"] for p in passages}
    parts = re.split(r"(=== STEP \d+ ===)", text)
    n_tags, step = 0, None
    for i, part in enumerate(parts):
        m = re.fullmatch(r"=== STEP (\d+) ===", part)
        if m:
            step = int(m.group(1)); continue
        if step is None:
            continue
        def sub(mm, channel):
            nonlocal n_tags
            n = pid.get((step, channel))
            if n is None:
                return mm.group(0)
            n_tags += 1
            return f"{mm.group(1)} [P{n}]"
        part = re.sub(r"(--- REASONING)", lambda mm: sub(mm, "reasoning"), part)
        part = re.sub(r"(--- ASSISTANT)", lambda mm: sub(mm, "content"), part)
        part = re.sub(r"(--- COMMAND \d+)", lambda mm: sub(mm, "tool_calls"), part)
        parts[i] = part
    return "".join(parts), n_tags


def judge_input(r):
    # the header's task: line names the split (e.g. lcb_conflicting_...), which would bias contradiction_failure
    text = re.sub(r"^task: .*$", "task: (withheld)", r["judge_text"], count=1, flags=re.M)
    text, _ = tag_passages(text, r["passages"])
    return "The transcript follows.\n\n" + text


def call_openai(model, text, effort):
    from openai import OpenAI
    r = OpenAI(timeout=900, max_retries=0).responses.create(model=model, instructions=RUBRIC, input=text, reasoning={"effort": effort},
                                  max_output_tokens=64000,
                                  text={"format": {"type": "json_schema", "name": "rows", "schema": SCHEMA, "strict": True}})
    u = r.usage.model_dump()
    if r.status != "completed":
        raise RuntimeError(f"status {r.status}")
    return json.loads(r.output_text), {"in": u["input_tokens"], "out": u["output_tokens"]}


def call_openrouter(model, text, effort):
    body = {"model": model, "messages": [{"role": "system", "content": RUBRIC}, {"role": "user", "content": text}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "rows", "strict": True, "schema": SCHEMA}},
            "reasoning": {"effort": effort}, "max_tokens": 64000, "usage": {"include": True},
            "provider": {"require_parameters": True,
                         **({"order": ["anthropic", "google-vertex", "azure"]} if model.startswith("anthropic/") else {})}}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                                          "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    if "error" in r:
        raise RuntimeError(str(r["error"])[:300])
    u = r.get("usage", {})
    return json.loads(r["choices"][0]["message"]["content"]), {"in": u.get("prompt_tokens", 0),
                                                               "out": u.get("completion_tokens", 0), "cost": u.get("cost")}


def label_rollout(r, judge, effort, tries=4):
    api, model, price = JUDGES[judge]
    fn = call_openai if api == "openai" else call_openrouter
    P = {p["n"]: p for p in r["passages"]}
    usage_all, err = [], None
    for t in range(tries):
        try:
            out, u = fn(model, judge_input(r), effort)
            usage_all.append(u)
            if [x["n"] for x in out["rows"]] == sorted(P):
                break
            err = f"row mismatch: {len(out['rows'])} rows for {len(P)} passages"
        except Exception as e:
            err = str(e)[:300]
            time.sleep(min(60, 5 * 2 ** t))
    else:
        return {"uid": r["uid"], "error": err, "usage": usage_all}
    cost = sum(u.get("cost") or (u["in"] * price[0] + u["out"] * price[1]) / 1e6 for u in usage_all)
    labels, withdrawn, bad_quotes = {}, 0, []
    for x in out["rows"]:
        p = P[x["n"]]
        good = [f for f in x["flags"] if quote_ok(f["quote"], p["text"])]
        withdrawn += len(x["flags"]) - len(good)
        bad_quotes += [{"n": x["n"], **f} for f in x["flags"] if f not in good]
        labels[x["n"]] = {L: any(f["label"] == L for f in good) for L in LABELS}
        labels[x["n"]]["subtypes"] = sorted({f["subtype"] for f in good})
        labels[x["n"]]["row_says"] = {L: x[L] for L in LABELS}
    return {"uid": r["uid"], "split": r["split"], "labels": labels, "withdrawn": withdrawn, "usage": usage_all,
            "cost": cost, "attempts": len(usage_all), "rationale": out["rationale"], "raw_rows": out["rows"],
            "bad_quotes": bad_quotes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--judge", choices=list(JUDGES), required=True)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--uids", required=True, help="file with one uid per line")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    load_env()
    want = set(open(a.uids).read().split("\n")) - {""}
    rolls = [json.loads(l) for l in gzip.open(ROLL, "rt")]
    rolls = [r for r in rolls if r["uid"] in want and r["passages"]]
    out = ROOT / "data" / "phase2" / a.run / a.judge
    out.mkdir(parents=True, exist_ok=True)

    def run(r):
        f = out / (re.sub(r"[^\w.-]", "_", r["uid"]) + ".json")
        if f.exists() and "labels" in json.loads(f.read_text()):
            return json.loads(f.read_text())
        rec = label_rollout(r, a.judge, a.effort)
        f.write_text(json.dumps(rec))
        return rec

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(run, rolls))
    ok = [x for x in recs if "labels" in x]
    print(f"{a.judge}: {len(ok)}/{len(recs)} rollouts labelled in {time.time() - t0:.0f}s, "
          f"cost ${sum(x['cost'] for x in ok):.3f}, withdrawn flags {sum(x['withdrawn'] for x in ok)}, "
          f"retries {sum(x['attempts'] - 1 for x in ok)}")
    for x in recs:
        if "labels" not in x:
            print("  FAILED", x["uid"], x["error"])


if __name__ == "__main__":
    main()
