#!/bin/bash
# Hard GPU release: at 06:52 UTC kill every process of ours on the GPUs (only our own; GPUs 0-3 are another user's).
cd /data/agastyas/rh_rl
KILL_AT=1791960720   # 2026-10-14 06:52:00 UTC
while [ "$(date +%s)" -lt "$KILL_AT" ]; do sleep 20; done
echo "watchdog firing $(date)"
for s in rl_main rl_vllm0 rl_vllm1; do tmux kill-session -t $s 2>/dev/null; done
sleep 20
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  [ "$(ps -o user= -p $pid 2>/dev/null | tr -d ' ')" = "agastyas" ] && kill -9 $pid && echo "killed $pid"
done
sleep 10
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while read l; do pid=${l%%,*}; echo "$l $(ps -o user= -p $pid)"; done
echo "watchdog done $(date)"
