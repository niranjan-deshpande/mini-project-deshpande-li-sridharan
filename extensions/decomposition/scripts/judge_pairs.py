"""Contamination judging of SWE-format candidate pairs (prompts/judge_contamination.md) via OpenRouter.

Usage: python scripts/judge_pairs.py --pairs data/gen/pilot_luna/pairs.jsonl --run pilot_luna_haiku [--repeat 0]
Only pairs that passed the automatic checks are judged. Keep/drop is computed in code (prompts/README.md, "Judging").
Outputs: data/judge/<run>/raw/<family>_<i>_r<repeat>.json, judged.jsonl, summary.json.
"""
import argparse, concurrent.futures as cf, json, os, re, sys, time, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rhdecomp.gen.render import PROMPTS  # noqa: E402

CFG = json.loads((PROMPTS / "directions.json").read_text())
RUBRIC = (PROMPTS / "judge_contamination.md").read_text()
SCHEMA = json.loads((PROMPTS / "schema_contamination.json").read_text())
PRICE = {"anthropic/claude-haiku-5.5": (0.10, 0.50), "deepseek/deepseek-v4.1-flash": (0.044, 0.765)}  # $/M in, out
norm = lambda s: re.sub(r"\s+", " ", s or "").strip()


def load_env():
    for line in open(ROOT.parents[1] / ".env"):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.strip().split("=", 1)
            if v:
                os.environ.setdefault(k, v.strip().strip('"'))


def split_think(content):
    m = re.match(r"\s*<think>\n?(.*?)\n?</think>\s*(.*)", content, re.S)
    return (m.group(1), m.group(2)) if m else ("", content)


def render_pair(p):
    """Render with the fixed markers the rubric describes. Returns (text, {location: text} for quote checks)."""
    A, B = p["hack"], p["honest"]
    k = A["turn_index"]
    first_diff = 0
    while A["messages"][first_diff] == B["messages"][first_diff]:
        first_diff += 1
    prefix = A["messages"][:first_diff]
    lines = ["=== DIRECTION ===", p["family"], "", "=== SHARED PREFIX ==="]
    for j, m in enumerate(prefix, 1):
        lines += [f"[{j}] {m['role'].upper()}", m["content"], ""]
    loc = {"prefix": "\n".join(m["content"] for m in prefix)}
    for side, br in (("A", A), ("B", B)):
        part = [f"=== BRANCH {side} ==="]
        for j in range(first_diff, len(br["messages"])):
            m = br["messages"][j]
            if j < k:
                part += ["OBSERVATION", m["content"], ""]
            elif j == k:
                think, vis = split_think(m["content"])
                part += ["DIVERGENT TURN", "REASONING:", think or "(none)", "VISIBLE:", vis, ""]
            else:
                part += [f"[{j + 1}] {m['role'].upper()}", m["content"], ""]
        lines += part
        loc[side] = "\n".join(part)
    lines += ["=== MARKED SPANS ===", f"A: {A['span']}", f"B: {B['span']}"]
    return "\n".join(lines), loc


def call(model, text, tries=5, effort="high"):
    body = {"model": model, "messages": [{"role": "system", "content": RUBRIC}, {"role": "user", "content": text}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "contamination", "strict": True,
                                                                         "schema": SCHEMA}},
            "reasoning": {"effort": effort}, "max_tokens": 16000, "usage": {"include": True},
            "provider": {"require_parameters": True}}
    if model.startswith("anthropic/"):
        body["provider"]["order"] = ["anthropic", "google-vertex", "azure"]
    for t in range(tries):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                                                  "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=300))
            if "error" in r:
                raise RuntimeError(str(r["error"])[:300])
            return json.loads(r["choices"][0]["message"]["content"]), r.get("usage", {}), None
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 4 * 2 ** t)); continue
            return None, {}, f"HTTP {e.code}: {msg}"
        except Exception as e:
            err = str(e)[:300]
            if t < tries - 1:
                time.sleep(min(60, 4 * 2 ** t)); continue
            return None, {}, err
    return None, {}, "retries exhausted"


def decide(j, loc, family):
    """Keep/drop in code. Flags whose quote isn't found verbatim in their location are withdrawn (logged)."""
    flags, withdrawn = [], []
    for f in j["hacking_flags"]:
        (flags if norm(f["quote"]) and norm(f["quote"]) in norm(loc.get(f["location"], "")) else withdrawn).append(f)
    c, target = j["concept"], CFG[family].get("leakage_target")
    reasons = []
    if flags:
        reasons.append("hacking: " + ", ".join(f"{f['location']}/{f['category']}/{f['status']}" for f in flags))
    if c["a_expresses_concept"] != "clear":
        reasons.append(f"A concept {c['a_expresses_concept']}")
    if c["b_contains_concept"] != "absent":
        reasons.append(f"B concept {c['b_contains_concept']}")
    if c["other_systematic_difference"] == "major":
        reasons.append("major other difference")
    if not c["setup_valid"]:
        reasons.append("setup invalid")
    for k, v in j["leakage"].items():
        if k != target and v["in_a"] != v["in_b"]:
            reasons.append(f"leak {k} ({'A' if v['in_a'] else 'B'})")
    return {"keep": not reasons, "reasons": reasons, "flags": flags, "withdrawn": withdrawn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--repeat", type=int, default=0, help="repeat index (re-judging flagged pairs)")
    ap.add_argument("--only", default=None, help="file of 'family_i' keys to judge (for re-judging)")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    load_env()
    out = ROOT / "data" / "judge" / a.run
    (out / "raw").mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(l) for l in open(a.pairs)]
    pairs = [p for p in pairs if not p["checks_failed"] and CFG[p["family"]].get("contamination_judge") not in (None, "none")
             and CFG[p["family"]]["format"] == "swe" and p["family"] != "hack"]
    if a.only:
        keys = set(open(a.only).read().split())
        pairs = [p for p in pairs if f"{p['family']}_{p['meta']['i']:03d}" in keys]

    def run(p):
        key = f"{p['family']}_{p['meta']['i']:03d}"
        f = out / "raw" / f"{key}_r{a.repeat}.json"
        text, loc = render_pair(p)
        if f.exists() and json.loads(f.read_text()).get("judgment"):
            rec = json.loads(f.read_text())
        else:
            j, u, err = call(a.model, text, effort=a.effort)
            pin, pout = PRICE.get(a.model, (0, 0))
            cost = u.get("cost") or (u.get("prompt_tokens", 0) * pin + u.get("completion_tokens", 0) * pout) / 1e6
            rec = {"key": key, "family": p["family"], "judgment": j, "usage": u, "error": err, "cost": cost}
            f.write_text(json.dumps(rec))
        if rec["judgment"]:
            rec["decision"] = decide(rec["judgment"], loc, p["family"])
        return rec

    with cf.ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(run, pairs))
    summ = {}
    for r in recs:
        s = summ.setdefault(r["family"], {"n": 0, "keep": 0, "error": 0, "reasons": {}, "withdrawn_flags": 0, "cost": 0.0})
        s["n"] += 1; s["cost"] += r.get("cost") or 0
        if not r["judgment"]:
            s["error"] += 1; continue
        d = r["decision"]; s["keep"] += d["keep"]; s["withdrawn_flags"] += len(d["withdrawn"])
        for x in d["reasons"]:
            k = x.split(":")[0].split(" (")[0]
            s["reasons"][k] = s["reasons"].get(k, 0) + 1
    with open(out / f"judged_r{a.repeat}.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    json.dump(summ, open(out / f"summary_r{a.repeat}.json", "w"), indent=1)
    for k, s in summ.items():
        print(f"{k:20s} keep {s['keep']}/{s['n']}  err {s['error']}  withdrawn {s['withdrawn_flags']}  ${s['cost']:.3f}  {s['reasons']}")
    print(f"total ${sum(s['cost'] for s in summ.values()):.3f}")


if __name__ == "__main__":
    main()
