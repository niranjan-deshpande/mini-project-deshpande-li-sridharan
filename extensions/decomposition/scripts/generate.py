"""Generate minimal pairs with an OpenAI model (Responses API, strict structured outputs), then run the automatic checks.

Usage: python scripts/generate.py --run pilot_luna --model gpt-6-luna --effort high --dirs hack eval_awareness --n 20
Raw responses are cached per pair in data/gen/<run>/raw/, so reruns only call the API for missing pairs.
Outputs: data/gen/<run>/pairs.jsonl (assembled, pairs_full.jsonl format, with check results), summary.json.
"""
import argparse, concurrent.futures as cf, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rhdecomp.gen.render import load_config, render, PROMPTS  # noqa: E402
from rhdecomp.gen.checks import check_swe, check_gc, to_pairs_format  # noqa: E402

PRICE = {"gpt-6-luna": (0.10, 0.01, 0.50), "gpt-5.6-terra": (2.00, 0.20, 12.00)}  # $/M in, cached in, out


def load_env():
    for line in open(ROOT.parents[1] / ".env"):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.strip().split("=", 1)
            if v:
                os.environ.setdefault(k, v.strip().strip('"'))


def cost(model, u):
    pin, pcache, pout = PRICE[model]
    cached = (u.get("input_tokens_details") or {}).get("cached_tokens", 0)
    return ((u["input_tokens"] - cached) * pin + cached * pcache + u["output_tokens"] * pout) / 1e6


def call(client, model, effort, prompt, schema, max_out, tries=5):
    for t in range(tries):
        try:
            r = client.responses.create(
                model=model, input=prompt, reasoning={"effort": effort}, max_output_tokens=max_out,
                text={"format": {"type": "json_schema", "name": "pair", "schema": schema, "strict": True}})
            u = r.usage.model_dump()
            if r.status != "completed":
                return None, u, f"status {r.status}: {getattr(r, 'incomplete_details', None)}"
            return json.loads(r.output_text), u, None
        except Exception as e:  # rate limits / transient errors
            err = str(e)[:300]
            if any(x in err.lower() for x in ("429", "500", "502", "503", "timeout", "timed out", "connection", "overloaded")):
                time.sleep(min(60, 4 * 2 ** t)); continue
            return None, {}, err
    return None, {}, "retries exhausted"


REPAIR = """

## Your previous answer
{prev}

## It failed these automatic checks
{fails}

Return the full corrected JSON object. Change only what is needed to fix these failures and keep everything else \
identical. For length failures, trim the longer side or add neutral detail to the shorter side so the ratio is at most 1.15."""


def run_checks(obj, d, meta, cfg, seed_by_id):
    try:
        return check_gc(obj, d) if cfg[d]["format"] == "gc" else check_swe(obj, d, meta, seed_by_id.get(meta.get("seed_id")))
    except Exception as e:
        return [f"check crashed: {type(e).__name__}: {e}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-output-tokens", type=int, default=32000)
    ap.add_argument("--no-repair", dest="repair", action="store_false", help="skip the one-round repair call")
    ap.add_argument("--budget", type=float, default=5.0, help="stop submitting new calls once spend exceeds this ($)")
    a = ap.parse_args()
    load_env()
    from openai import OpenAI
    client = OpenAI(timeout=480, max_retries=0)  # fail fast on dead connections (sleep/wifi); call() retries

    cfg, seeds = load_config()
    seed_by_id = {s["id"]: s for s in seeds}
    out = ROOT / "data" / "gen" / a.run
    (out / "raw").mkdir(parents=True, exist_ok=True)

    jobs = []
    for d in a.dirs:
        rng = np.random.default_rng(int(hashlib.md5(d.encode()).hexdigest(), 16) % 2**32)  # same seeds for any model
        uses = {}
        schema = json.loads((PROMPTS / cfg[d]["schema"]).read_text())
        for i in range(a.n):
            prompt, meta = render(d, i, cfg, seeds, uses, rng)
            jobs.append((d, i, prompt, meta, schema))

    spent = [0.0]

    def run(job):
        d, i, prompt, meta, schema = job
        f = out / "raw" / f"{d}_{i:03d}.json"
        if f.exists():
            rec = json.loads(f.read_text())
            if rec.get("obj") is not None:
                return rec
        if spent[0] > a.budget:
            return {"meta": meta, "obj": None, "usage": {}, "error": "budget stop", "cost": 0}
        obj, u, err = call(client, a.model, a.effort, prompt, schema, a.max_output_tokens)
        c = cost(a.model, u) if u else 0.0
        spent[0] += c
        rec = {"meta": meta, "model": a.model, "effort": a.effort, "obj": obj, "usage": u, "error": err, "cost": c,
               "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16]}
        if obj is not None and a.repair:
            first = run_checks(obj, d, meta, cfg, seed_by_id)
            rec["first_pass_fails"] = first
            if first and not any(x.startswith("check crashed") for x in first):
                fixed, u2, err2 = call(client, a.model, a.effort, prompt + REPAIR.format(
                    prev=json.dumps(obj, indent=1), fails="\n".join(f"- {x}" for x in first)), schema, a.max_output_tokens)
                c2 = cost(a.model, u2) if u2 else 0.0
                spent[0] += c2; rec["cost"] += c2
                rec["repair"] = {"usage": u2, "error": err2, "cost": c2}
                if fixed is not None:
                    rec["obj_first"], rec["obj"] = obj, fixed
        f.write_text(json.dumps(rec))
        return rec

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(run, jobs))
    print(f"{len(recs)} calls in {time.time() - t0:.0f}s, new spend ${spent[0]:.3f}")
    # pairs.jsonl / summary cover every pair saved in this run directory, not just this invocation's jobs
    recs = [json.loads(f.read_text()) for f in sorted((out / "raw").glob("*.json"))]

    summary, rows = {}, []
    for rec in recs:
        m, d = rec["meta"], rec["meta"]["direction"]
        s = summary.setdefault(d, {"n": 0, "api_fail": 0, "pass": 0, "fail_reasons": {}, "cost": 0.0,
                                    "out_tokens": [], "reasoning_tokens": []})
        s["n"] += 1; s["cost"] += rec.get("cost", 0)
        if rec["usage"]:
            s["out_tokens"].append(rec["usage"]["output_tokens"])
            s["reasoning_tokens"].append((rec["usage"].get("output_tokens_details") or {}).get("reasoning_tokens", 0))
        if rec["obj"] is None:
            s["api_fail"] += 1; s["fail_reasons"].setdefault(f"api: {rec['error'][:60]}", 0)
            s["fail_reasons"][f"api: {rec['error'][:60]}"] += 1
            continue
        obj = rec["obj"]
        fails = run_checks(obj, d, m, cfg, seed_by_id)
        s["pass"] += not fails
        if "first_pass_fails" in rec:
            s["first_pass_pass"] = s.get("first_pass_pass", 0) + (not rec["first_pass_fails"])
        for x in fails:
            key = x.split(":")[0]
            s["fail_reasons"][key] = s["fail_reasons"].get(key, 0) + 1
        row = ({"family": d, "gc": obj, "meta": m} if cfg[d]["format"] == "gc" else to_pairs_format(obj, d, m)) \
            if not any(x.startswith("check crashed") for x in fails) else {"family": d, "raw": obj, "meta": m}
        row.update({"checks_failed": fails, "plan": obj.get("plan")})
        rows.append(row)
    for s in summary.values():
        s["pass_rate"] = s["pass"] / s["n"]
        for k in ("out_tokens", "reasoning_tokens"):
            s[k] = {"mean": float(np.mean(s[k])) if s[k] else None, "max": int(max(s[k])) if s[k] else None}
    with open(out / "pairs.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    tot = sum(s["cost"] for s in summary.values())
    for d, s in summary.items():
        print(f"{d:24s} first-pass {s.get('first_pass_pass', '-')}/{s['n']}  final pass {s['pass']}/{s['n']}  api_fail {s['api_fail']}  ${s['cost']:.3f}  "
              f"out {s['out_tokens']['mean']}  reasoning {s['reasoning_tokens']['mean']}  {s['fail_reasons']}")
    print(f"total cost of all records in this run: ${tot:.3f}")


if __name__ == "__main__":
    main()
