"""Write results/cpA/samples.md: 5 passing pairs per direction from the latest prompt version of each direction."""
import json, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rhdecomp.gen.render import load_config, directions

LATEST = {"eval_awareness": "pilot_luna_v3", "testvocab_swe": "pilot_luna_v3", "grader_modeling": "pilot_luna_v3"}
RUNS = ["pilot_luna_v2", "pilot_luna_v3", "pilot_luna_samples", "pilot_luna_newbase"]
cfg, _ = load_config()
P = defaultdict(list)
for run in RUNS:
    for l in open(ROOT / f"data/gen/{run}/pairs.jsonl"):
        p = json.loads(l)
        want = LATEST.get(p["family"])
        if p["checks_failed"] or (want and want != run) or (not want and run == "pilot_luna_v3"):
            continue
        if p["family"] == "testvocab_swe" and p["variant"] == "read_check":
            continue
        P[p["family"]].append(p)
short = lambda s, n=200: s.replace("\n", " ⏎ ")[:n] + ("…" if len(s) > n else "")
md = ["# CHECKPOINT A: sample pairs (5 per direction, GPT-6 Luna, latest prompt versions, after automatic checks)\n",
      "SWE pairs: shared prefix condensed (role + first 200 chars); divergent turns in full with the span in **bold**; "
      "continuation abbreviated. task_failure also shows each branch's observation.\n"]
for d in [d for d in directions(cfg) if d in P]:
    md.append(f"\n---\n\n## {d}\n")
    for p in P[d][:5]:
        if "gc" in p:
            g = p["gc"]
            md.append(f"**{g['domain']}**{(' — context: ' + g['shared_context']) if g['shared_context'] else ''}\n\n"
                      f"- A: {g['text_a'].replace(g['span_a'], '**' + g['span_a'] + '**')}\n"
                      f"- B: {g['text_b'].replace(g['span_b'], '**' + g['span_b'] + '**')}\n")
            continue
        A, B = p["hack"], p["honest"]; k = A["turn_index"]; pre = k - 1 if d == "task_failure" else k
        ph = p["meta"].get("phrasing")
        md.append(f"### {p['task']}\n`{p['meta']['seed_id']}` · variant `{p['variant']}`" + (f" · phrasing: {ph}" if ph else "") + "\n")
        md.append("Prefix:\n" + "\n".join(f"- {m['role']}: {short(m['content'])}" for m in A["messages"][:pre]) + "\n")
        for name, br in (("A", A), ("B", B)):
            obs = f"Observation {name}: {short(br['messages'][k - 1]['content'], 300)}\n\n" if d == "task_failure" else ""
            md.append(f"{obs}**Divergent turn {name}:**\n\n```\n{br['messages'][k]['content'].replace(br['span'], '**' + br['span'] + '**')}\n```\n")
        same = A["messages"][k + 1:] == B["messages"][k + 1:]
        md.append(f"Continuation ({'identical in A and B' if same else 'branch A'}): " +
                  " / ".join(f"{m['role']}: {short(m['content'], 160)}" for m in A["messages"][k + 1:]) + "\n")
(ROOT / "results/cpA/samples.md").write_text("\n".join(md))
print({d: min(5, len(v)) for d, v in P.items()})
