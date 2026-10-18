#!/bin/bash
# Probe scoring of the step-1 pilot (low + medium) on GPUs 5-7; a DP1 gpt-oss server stays on GPU 4.
cd /data/agastyas/rh_rl
cat p1_gptoss_low.jsonl p1_gptoss_medium.jsonl > p1_gptoss_all.jsonl
source /data/agastyas/rh_probe/env.sh
export HF_HUB_OFFLINE=0 HF_HOME=/data/agastyas/rh_probe/hf_home PYTHONPATH=/data/agastyas/rh_probe/pylib TMPDIR=/data/agastyas/rh_probe/tmp TRITON_CACHE_DIR=/data/agastyas/rh_probe/tmp/triton
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((5 + i)) $PY score_probe_gptoss.py p1_gptoss_all.jsonl pilot_tasks.jsonl calib_gptoss --shard $i/3 > logs/calib_gptoss_$i.log 2>&1 &
done
wait; echo "ALL DONE $(date)"
