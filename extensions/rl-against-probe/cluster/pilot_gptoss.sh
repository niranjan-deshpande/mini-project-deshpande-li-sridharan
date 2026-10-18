#!/bin/bash
# Step 1 pilot on TRAIN tasks (doubles as probe-calibration data): 40 per split x n=8, efforts low and medium.
cd /data/agastyas/rh_rl
venv/bin/python - <<'PY'
import json, random
T = [json.loads(l) for l in open("train_tasks.jsonl")]
random.Random(1).shuffle(T)
out = []
for s in ["original", "oneoff", "conflicting"]:
    out += [t for t in T if t["split"] == s][:40]
open("pilot_tasks.jsonl", "w").write("".join(json.dumps(t) + "\n" for t in out))
print(len(out))
PY
for i in $(seq 1 120); do curl -s -o /dev/null -H "Authorization: Bearer $(cat /data/agastyas/rh_probe/api_key)" http://127.0.0.1:8400/v1/models && break; sleep 10; done
echo "server up $(date)"
for e in low medium; do
  venv/bin/python sample.py pilot_tasks.jsonl p1_gptoss_$e.jsonl --chat gpt-oss-120b --mode lenient --effort $e --n 8 --workers 120 > logs/p1_gptoss_$e.log 2>&1 &
done
wait; echo "ALL DONE $(date)"
