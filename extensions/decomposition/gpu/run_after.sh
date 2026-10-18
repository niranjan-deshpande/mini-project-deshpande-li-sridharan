#!/bin/bash
# Replaces run_pod.sh's tail: wait for the running session-1 python to exit, run session 2 (emotions), then delete the
# pod (20-min grace on failure; touch /workspace/HOLD to keep it).
set -u
W=/workspace; POD_ID=${POD_ID:?}; S1PID=${S1PID:?}
while kill -0 $S1PID 2>/dev/null; do sleep 20; done
echo "[$(date +%T)] session 1 exited"
rc=0
if grep -q "SESSION1 COMPLETE" $W/out/session1.log; then
  cd $W/code && timeout 2h $W/venv/bin/python gpu/session2_emotions.py --model $W/models/Qwen3.8-27B --data $W/data --out $W/out; rc=$?
else rc=1; fi
echo "[$(date +%T)] session 2 exit code $rc"
if [ "$rc" != "0" ]; then
  echo "failed; waiting 20 min before deleting (touch $W/HOLD to keep)"
  for i in $(seq 1 40); do sleep 30; [ -f $W/HOLD ] && { echo "HOLD found; not deleting"; exit 0; }; done
fi
KEY=$(grep -i '^apikey' /root/.runpod/config.toml | head -1 | sed -E "s/^[^=]*= *//; s/^[\"']//; s/[\"']$//")
for i in 1 2 3 4 5; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "https://rest.runpod.io/v1/pods/$POD_ID" -H "Authorization: Bearer $KEY")
  echo "delete -> $code"; [ "$code" = "200" ] || [ "$code" = "204" ] && break; sleep 20
done
