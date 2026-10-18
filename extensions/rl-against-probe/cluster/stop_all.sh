#!/bin/bash
for s in rl_main rl_armB rl_vllm0 rl_vllm1; do tmux kill-session -t $s 2>/dev/null; done
sleep 15
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  [ "$(ps -o user= -p $pid 2>/dev/null | tr -d " ")" = "agastyas" ] && kill -9 $pid && echo "killed $pid"
done
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
