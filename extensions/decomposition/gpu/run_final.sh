#!/bin/bash
# Wait for session 2 to exit, then wait for the laptop to signal it has downloaded outputs (touch /workspace/DOWNLOADED)
# or 40 minutes, whichever first; then delete the pod. touch /workspace/HOLD to keep it.
set -u
W=/workspace; POD_ID=${POD_ID:?}; S2PID=${S2PID:?}
while kill -0 $S2PID 2>/dev/null; do sleep 20; done
echo "[$(date +%T)] session 2 exited"
for i in $(seq 1 80); do [ -f $W/DOWNLOADED ] && break; [ -f $W/HOLD ] && { echo HOLD; exit 0; }; sleep 30; done
KEY=$(grep -i '^apikey' /root/.runpod/config.toml | head -1 | sed -E "s/^[^=]*= *//; s/^[\"']//; s/[\"']$//")
for i in 1 2 3 4 5; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "https://rest.runpod.io/v1/pods/$POD_ID" -H "Authorization: Bearer $KEY")
  echo "[$(date +%T)] delete -> $code"; [ "$code" = "200" ] || [ "$code" = "204" ] && break; sleep 20
done
