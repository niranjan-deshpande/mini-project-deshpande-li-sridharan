"""Resumable driver for the bulk run: generate -> dedupe/top-up -> contamination judging -> Phase 2 labels -> final sets.

Every stage skips work already saved on disk, so after an interruption just rerun:
    caffeinate -i .venv/bin/python scripts/bulk.py
Progress goes to data/bulk.log; stage markers to data/bulk_status.json.
"""
import collections, gzip, hashlib, json, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/bin/python")
RUN = "bulk"
GEN = ROOT / "data/gen" / RUN
JUDGE = ROOT / "data/judge" / f"{RUN}_haiku"
P2 = ROOT / "data/phase2" / RUN
FINAL = ROOT / "data/final"
CFG = json.loads((ROOT / "prompts/directions.json").read_text())
DIRS = [k for k, v in CFG.items() if isinstance(v, dict) and "prompt" in v]
OVERSHOOT = 1.15          # generate 15% extra to cover check failures, duplicates and contamination drops
MAX_SPAN_REPEATS = 2      # an identical (normalised) branch-A span may appear at most this many times per direction
GEN_BUDGET = 30.0         # $ cap per generate.py invocation
LOG = open(ROOT / "data/bulk.log", "a", buffering=1)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); LOG.write(line + "\n")


def sh(args):
    log("$ " + " ".join(args[1:]))
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-30:])
    LOG.write(tail + "\n")
    if r.returncode:
        log(f"command failed ({r.returncode}):\n{tail}"); sys.exit(1)
    return r.stdout


def generate(dirs_n):
    """dirs_n: {direction: n}; one generate.py call per distinct n."""
    by_n = collections.defaultdict(list)
    for d, n in dirs_n.items():
        by_n[n].append(d)
    for n, ds in sorted(by_n.items()):
        sh([PY, "scripts/generate.py", "--run", RUN, "--model", "gpt-6-luna", "--effort", "high", "--dirs", *ds,
            "--n", str(n), "--workers", "64", "--budget", str(GEN_BUDGET)])


def load_pairs():
    return [json.loads(l) for l in open(GEN / "pairs.jsonl")]


def span_key(p):
    s = p["hack"]["span"] if "hack" in p else p["gc"]["span_a"]
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def usable(pairs):
    """Passing pairs per direction, with exact-duplicate branch-A spans beyond MAX_SPAN_REPEATS removed (lowest index kept)."""
    out, dups = collections.defaultdict(list), collections.Counter()
    seen = collections.Counter()
    for p in sorted(pairs, key=lambda p: (p["family"], p["meta"]["i"])):
        if p["checks_failed"]:
            continue
        k = (p["family"], span_key(p))
        seen[k] += 1
        if seen[k] > MAX_SPAN_REPEATS:
            dups[p["family"]] += 1; continue
        out[p["family"]].append(p)
    return out, dups


def stage_generate():
    target = {d: CFG[d]["n"] for d in DIRS}
    n_first = {d: int(round(target[d] * OVERSHOOT)) for d in DIRS}
    generate(n_first)
    for rnd in range(3):  # top up directions that fell short after checks + dedupe
        good, dups = usable(load_pairs())
        short = {d: target[d] - len(good[d]) for d in DIRS if len(good[d]) < target[d]}
        log(f"usable per direction: {{{', '.join(f'{d}: {len(good[d])}' for d in DIRS)}}}; duplicates removed {dict(dups)}")
        if not short:
            break
        cur = collections.Counter(p["family"] for p in load_pairs())
        generate({d: cur[d] + int(short[d] * 1.5) + 2 for d in short})


def stage_judge():
    pairs_path = str(GEN / "pairs.jsonl")
    sh([PY, "scripts/judge_pairs.py", "--pairs", pairs_path, "--run", f"{RUN}_haiku", "--workers", "24"])
    flagged = [json.loads(l)["key"] for l in open(JUDGE / "judged_r0.jsonl")
               if json.loads(l).get("decision") and not json.loads(l)["decision"]["keep"]]
    (JUDGE / "flagged.txt").write_text("\n".join(flagged))
    log(f"contamination r0: {len(flagged)} flagged")
    for r in (1, 2):
        sh([PY, "scripts/judge_pairs.py", "--pairs", pairs_path, "--run", f"{RUN}_haiku", "--repeat", str(r),
            "--only", str(JUDGE / "flagged.txt"), "--workers", "24"])


def contamination_drops():
    """Keys flagged in >= 2 of 3 runs (r0 + two re-judges); errors count as not flagged."""
    votes = collections.Counter()
    for r in (0, 1, 2):
        f = JUDGE / f"judged_r{r}.jsonl"
        if f.exists():
            for l in open(f):
                x = json.loads(l)
                if x.get("decision") and not x["decision"]["keep"]:
                    votes[x["key"]] += 1
    return {k for k, v in votes.items() if v >= 2}


def stage_phase2():
    rolls = [json.loads(l) for l in gzip.open(ROOT.parents[1] / "reproduction/multi-model/results/rollouts/qwen_rollouts.jsonl.gz", "rt")]
    uids = [r["uid"] for r in rolls if r["split"] in ("conflicting", "oneoff") and r["passages"]]
    P2.mkdir(parents=True, exist_ok=True)
    (P2 / "uids.txt").write_text("\n".join(uids) + "\n")
    # reuse pilot2 labels (same rubric, same code) for rollouts already labelled
    for sub in ("luna", "haiku", "luna_evalsnip", "haiku_evalsnip"):
        src, dst = ROOT / "data/phase2/pilot2" / sub, P2 / sub
        dst.mkdir(exist_ok=True)
        for f in src.glob("*.json"):
            if not (dst / f.name).exists() and json.loads(f.read_text()).get("split") in ("conflicting", "oneoff"):
                shutil.copy(f, dst / f.name)
    for judge in ("luna", "haiku"):
        sh([PY, "scripts/label_phase2.py", "--run", RUN, "--judge", judge, "--uids", str(P2 / "uids.txt"), "--workers", "12"])
        sh([PY, "scripts/label_eval_focus.py", "--run", RUN, "--judge", judge, "--uids", str(P2 / "uids.txt"), "--workers", "12"])


def stage_final():
    FINAL.mkdir(parents=True, exist_ok=True)
    good, dups = usable(load_pairs())
    drops = contamination_drops()
    manifest = {}
    for d in DIRS:
        keep = [p for p in good[d] if f"{d}_{p['meta']['i']:03d}" not in drops]
        n_drop = len(good[d]) - len(keep)
        with open(FINAL / f"pairs_{d}.jsonl", "w") as f:
            for p in keep:
                f.write(json.dumps(p) + "\n")
        manifest[d] = {"target": CFG[d]["n"], "final_n": len(keep), "contamination_dropped": n_drop,
                       "duplicate_spans_removed": dups[d]}
    # Phase 2 consensus labels per passage
    def load(sub):
        return {json.loads(f.read_text())["uid"]: json.loads(f.read_text()) for f in (P2 / sub).glob("*.json")}
    lu, ha, lue, hae = load("luna"), load("haiku"), load("luna_evalsnip"), load("haiku_evalsnip")
    narrow = {"explicit_evaluation", "task_design_inference"}
    rows = []
    for u in sorted(set(lu) & set(ha) & set(lue) & set(hae)):
        if "labels" not in lu[u] or "labels" not in ha[u] or "eval" not in lue[u] or "eval" not in hae[u]:
            continue
        for n in lu[u]["labels"]:
            c = (lu[u]["labels"][n]["contradiction_failure"], ha[u]["labels"][n]["contradiction_failure"])
            e = (lue[u]["eval"][n], hae[u]["eval"][n])
            en = (bool(set(lue[u]["subtypes"][n]) & narrow), bool(set(hae[u]["subtypes"][n]) & narrow))
            cons = lambda t: t[0] if t[0] == t[1] else None
            rows.append({"uid": u, "n": int(n), "split": lu[u]["split"],
                         "contradiction_failure": cons(c), "contradiction_votes": c,
                         "eval_awareness": cons(e), "eval_awareness_votes": e,
                         "eval_awareness_narrow": cons(en)})
    with open(FINAL / "phase2_labels.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    manifest["phase2"] = {"rollouts": len({r["uid"] for r in rows}), "passages": len(rows),
                          **{k: {"positive": sum(r[k] is True for r in rows), "negative": sum(r[k] is False for r in rows),
                                 "dropped_disagreement": sum(r[k] is None for r in rows)}
                             for k in ("contradiction_failure", "eval_awareness", "eval_awareness_narrow")}}
    json.dump(manifest, open(FINAL / "manifest.json", "w"), indent=1)
    log("final manifest:\n" + json.dumps(manifest, indent=1))


STAGES = [("generate", stage_generate), ("judge", stage_judge), ("phase2", stage_phase2), ("final", stage_final)]

if __name__ == "__main__":
    status_f = ROOT / "data/bulk_status.json"
    status = json.loads(status_f.read_text()) if status_f.exists() else {}
    only = sys.argv[1:]
    for name, fn in STAGES:
        if only and name not in only:
            continue
        if status.get(name) == "done" and not only:
            log(f"stage {name}: already done"); continue
        log(f"stage {name}: start")
        fn()
        status[name] = "done"; status_f.write_text(json.dumps(status, indent=1))
        log(f"stage {name}: done")
    log("ALL STAGES COMPLETE")
