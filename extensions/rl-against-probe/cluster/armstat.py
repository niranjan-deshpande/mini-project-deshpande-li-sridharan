"""Progress of an RL arm in blocks of steps: pass rate per split, probe fire rate at tau (computed offline, so it is
also defined for arm A), median analysis length, truncation. Usage: python armstat.py RUN [--tau 770537] [--block 5]"""
import argparse, glob, json, statistics as st
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--tau", type=float, default=770537)
ap.add_argument("--block", type=int, default=5); a = ap.parse_args()
files = sorted(glob.glob(f"{a.run}/samples/step_*.jsonl"))
L = [json.loads(l) for l in open(f"{a.run}/log.jsonl")]
print(f"{a.run}: {len(L)} steps, mean step time {st.mean(r['time'] for r in L):.0f}s, IS |dlp| {L[-1]['is_absdiff']:.3f}")
print("steps    imp_pass  orig_pass  fire@tau(imp)  fire@tau(orig)  med_analysis_chars  trunc")
for b in range(0, len(files), a.block):
    S = [json.loads(l) for f in files[b:b + a.block] for l in open(f)]
    imp = [s for s in S if s["split"] != "original"]; ori = [s for s in S if s["split"] == "original"]
    fr = lambda Q: sum(s["probe"] > a.tau for s in Q if s["probe"] == s["probe"]) / max(1, len(Q))
    print(f"{b + 1:3d}-{min(b + a.block, len(files)):3d}  {sum(s['pass'] for s in imp) / len(imp):8.2f}  {sum(s['pass'] for s in ori) / len(ori):9.2f}"
          f"  {fr(imp):13.2f}  {fr(ori):14.2f}  {st.median(len(s['analysis']) for s in S):18.0f}  {sum(s['finish'] == 'length' for s in S) / len(S):5.2f}")
