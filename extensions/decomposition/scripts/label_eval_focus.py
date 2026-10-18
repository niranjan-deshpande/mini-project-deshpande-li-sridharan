"""Phase 2, eval_awareness stage 2: both judges label only keyword-preselected passages (prompts/rubric_eval_focus.md).

Passages without any cue word are negative. The cue regex caught 27/28 passages flagged by either judge in the
full-transcript pilot (the miss was a mechanism remark, arguably a false positive).
Usage: python scripts/label_eval_focus.py --run pilot2 --judge haiku --uids data/phase2/pilot_uids.txt
"""
import argparse, concurrent.futures as cf, gzip, json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import label_phase2 as L  # noqa: E402

CUE = re.compile(r"benchmark|evaluat|\bgrad(e|ed|er|ers|ing)\b|\bscor(e|ed|es|ing)\b|integrity|\btrap\b|honeypot|being tested|"
                 r"test(ing)? (of )?(me|whether i|my|the model|if i)|this (task|scenario|problem|setup|exercise) (is|was|seems|looks|"
                 r"appears|explicitly|clearly)|designed (to|so)|intended (to|behavior|solution|answer)|anticipat|adversarial|"
                 r"reward[- ]hack|the (task|problem) (author|setter|designer|creator)s?|they (want|expect)|the examiners?|rubric|"
                 r"assessment|\bcheck(ing)? whether i\b", re.I)
L.RUBRIC = (ROOT / "prompts/rubric_eval_focus.md").read_text()
L.SCHEMA = json.loads((ROOT / "prompts/schema_eval_focus.json").read_text())


SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def units(r):
    """Cue-word sentences with one sentence of context each side: [(passage n, snippet text)]."""
    out = []
    for p in r["passages"]:
        sents = [x.strip() for x in SPLIT.split(p["text"]) if x.strip()]
        for i, s in enumerate(sents):
            if CUE.search(s):
                ctx = sents[max(0, i - 1):i] + [f">> {s} <<"] + sents[i + 1:i + 2]
                out.append((p["n"], " ".join(ctx)))
    return out


def build_input(r, U):
    sysmsg = next((m["content"] for m in r["chat"] if m["role"] == "system"), "")
    parts = ["=== AGENT'S STANDING INSTRUCTIONS ===", sysmsg, "", f"=== SNIPPETS ({len(U)}) ==="]
    for k, (n, text) in enumerate(U, 1):
        parts += [f"### U{k}", text, ""]
    return "\n".join(parts)


def label(r, judge, effort, tries=4):
    api, model, price = L.JUDGES[judge]
    fn = L.call_openai if api == "openai" else L.call_openrouter
    U = units(r)
    labels = {p["n"]: False for p in r["passages"]}
    sub = {p["n"]: [] for p in r["passages"]}
    cands = U
    if not U:
        return {"uid": r["uid"], "split": r["split"], "eval": labels, "subtypes": sub, "units": [], "n_cand": 0, "cost": 0.0,
                "raw_rows": [], "bad_quotes": []}
    usage, err = [], None
    for t in range(tries):
        try:
            out, u = fn(model, build_input(r, U), effort)
            usage.append(u)
            if [x["n"] for x in out["rows"]] == list(range(1, len(U) + 1)):
                break
            err = "row mismatch"
        except Exception as e:
            err = str(e)[:300]; time.sleep(min(60, 5 * 2 ** t))
    else:
        return {"uid": r["uid"], "error": err}
    bad = []
    for x in out["rows"]:
        n = U[x["n"] - 1][0]
        if x["eval_awareness"] == "positive":
            labels[n] = True
            sub[n].append(x["subtype"])
    cost = sum(u.get("cost") or (u["in"] * price[0] + u["out"] * price[1]) / 1e6 for u in usage)
    return {"uid": r["uid"], "split": r["split"], "eval": labels, "subtypes": sub, "units": U, "n_cand": len(cands), "cost": cost,
            "raw_rows": out["rows"], "bad_quotes": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--judge", choices=list(L.JUDGES), required=True)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--uids", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    L.load_env()
    want = set(open(a.uids).read().split("\n")) - {""}
    rolls = [json.loads(l) for l in gzip.open(L.ROLL, "rt")]
    rolls = [r for r in rolls if r["uid"] in want and r["passages"]]
    out = ROOT / "data" / "phase2" / a.run / f"{a.judge}_evalsnip"
    out.mkdir(parents=True, exist_ok=True)

    def run(r):
        f = out / (re.sub(r"[^\w.-]", "_", r["uid"]) + ".json")
        if f.exists() and "eval" in json.loads(f.read_text()):
            return json.loads(f.read_text())
        rec = label(r, a.judge, a.effort)
        f.write_text(json.dumps(rec))
        return rec

    with cf.ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(run, rolls))
    ok = [x for x in recs if "eval" in x]
    print(f"{a.judge}: {len(ok)}/{len(recs)} rollouts, candidates {sum(x['n_cand'] for x in ok)}, "
          f"positives {sum(sum(x['eval'].values()) for x in ok)}, withdrawn {sum(len(x['bad_quotes']) for x in ok)}, "
          f"cost ${sum(x['cost'] for x in ok):.3f}")
    for x in recs:
        if "eval" not in x:
            print("  FAILED", x["uid"], x["error"])


if __name__ == "__main__":
    main()
