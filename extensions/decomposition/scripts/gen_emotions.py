"""Generate emotion stories (and neutral dialogues) with GPT-6 Luna, following Sofroniew et al. 2026 (notes/emotions_method.md).

One call per (emotion, topic) -> 12 stories. Cached per call; rerun to resume. Stories containing a banned form of their
emotion word are dropped (whole-word, case-insensitive).
Usage: python scripts/gen_emotions.py --run emo40 [--limit-topics 50] [--neutral]
"""
import argparse, concurrent.futures as cf, json, os, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate import load_env, call, cost  # noqa: E402

E = ROOT / "prompts/emotions"
CFG = json.loads((E / "emotions.json").read_text())
TOPICS = json.loads((E / "topics.json").read_text())


def strip_comment(t):
    return re.sub(r"<!--.*?-->\s*", "", t, flags=re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--emotions", default="subset_40")
    ap.add_argument("--limit-topics", type=int, default=50)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--neutral", action="store_true", help="generate neutral dialogues instead")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--only", nargs="*", help="restrict to these emotions (pilot)")
    a = ap.parse_args()
    load_env()
    from openai import OpenAI
    client = OpenAI(timeout=480, max_retries=0)
    out = ROOT / "data/emotions" / a.run
    (out / "raw").mkdir(parents=True, exist_ok=True)
    topics = TOPICS[: a.limit_topics]
    if a.neutral:
        tmpl = strip_comment((E / "gen_neutral_dialogues.md").read_text())
        schema = json.loads((E / "schema_neutral_dialogues.json").read_text())
        jobs = [("neutral", t) for t in topics]
    else:
        tmpl = strip_comment((E / "gen_emotion_stories.md").read_text())
        schema = json.loads((E / "schema_emotion_stories.json").read_text())
        emos = a.only or CFG[a.emotions]
        jobs = [(e, t) for e in emos for t in topics]
    spent = [0.0]

    def run(job):
        emo, t = job
        f = out / "raw" / f"{emo}_{t['id']}.json"
        if f.exists() and json.loads(f.read_text()).get("obj"):
            return json.loads(f.read_text())
        if spent[0] > a.budget:
            return {"emotion": emo, "topic_id": t["id"], "obj": None, "error": "budget stop"}
        gloss = CFG.get("glosses_optional", {}).get(emo)
        prompt = (tmpl.replace("{emotion}", emo).replace("{topic}", t["topic"]).replace("{topic_id}", t["id"])
                  .replace("{n_stories}", str(a.n)).replace("{n_dialogues}", str(a.n))
                  .replace("{gloss_line}", f"(Here '{emo}' means: {gloss}.)" if gloss else ""))
        prompt = re.sub(r"\[ours:[^\]]*\]", "", prompt).replace("[ours]", "")
        assert not re.search(r"\{[a-z_]+\}", prompt), prompt
        obj, u, err = call(client, "gpt-6-luna", a.effort, prompt, schema, 32000)
        c = cost("gpt-6-luna", u) if u else 0.0
        spent[0] += c
        rec = {"emotion": emo, "topic_id": t["id"], "obj": obj, "usage": u, "error": err, "cost": c}
        f.write_text(json.dumps(rec))
        return rec

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(run, jobs))
    rows, dropped, total = [], 0, 0
    for r in recs:
        if not r.get("obj"):
            continue
        items = r["obj"].get("dialogues") if a.neutral else r["obj"]["stories"]
        forms = [] if a.neutral else CFG["ban_forms"].get(r["emotion"], [r["emotion"]])
        rx = re.compile(r"\b(" + "|".join(map(re.escape, forms)) + r")\b", re.I) if forms else None
        for k, s in enumerate(items):
            total += 1
            if rx and rx.search(s["text"]):
                dropped += 1; continue
            rows.append({"emotion": r["emotion"], "topic_id": r["topic_id"], "k": k, **s})
    with open(out / "items.jsonl", "w") as fh:
        for x in rows:
            fh.write(json.dumps(x) + "\n")
    print(f"{len(recs)} calls in {time.time() - t0:.0f}s; items {len(rows)} kept / {total} "
          f"(dropped {dropped} naming the emotion); failed calls {sum(1 for r in recs if not r.get('obj'))}; "
          f"cost ${sum(r.get('cost', 0) for r in recs):.3f}")


if __name__ == "__main__":
    main()
