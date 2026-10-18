#!/bin/bash
# Backstop GPU release: at 07:15 UTC force-kill every process owned by agastyas that holds any GPU (any index).
# Finds them two ways: nvidia-smi compute apps, and /proc fds open on /dev/nvidia* (catches PIDs nvidia-smi can't map).
# Never touches other users' processes.
cd "$(dirname "$0")"
KILL_AT=1791962100   # 2026-10-14 07:15:00 UTC
ME=agastyas
gpu_pids() {
  { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null
    for d in /proc/[0-9]*; do
      [ "$(stat -c %U "$d" 2>/dev/null)" = "$ME" ] || continue
      ls -l "$d/fd" 2>/dev/null | grep -q '/dev/nvidia[0-9]' && echo "${d#/proc/}"
    done; } | sort -u | while read pid; do
      [ -n "$pid" ] && [ "$pid" != "$$" ] && [ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" = "$ME" ] && echo "$pid"
    done
}
if [ "$1" = "--dry" ]; then echo "would kill: $(gpu_pids | tr '\n' ' ')"; exit 0; fi
while [ "$(date +%s)" -lt "$KILL_AT" ]; do sleep 20; done
echo "watchdog2 firing $(date -u)"
for attempt in 1 2 3 4 5; do
  P=$(gpu_pids)
  [ -z "$P" ] && { echo "attempt $attempt: no GPU processes of $ME"; break; }
  for pid in $P; do echo "attempt $attempt: kill -9 $pid $(ps -o args= -p $pid | cut -c1-120)"; kill -9 "$pid"; done
  sleep 15
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "remaining GPU processes of $ME: $(gpu_pids | tr '\n' ' ')"
echo "watchdog2 done $(date -u)"
