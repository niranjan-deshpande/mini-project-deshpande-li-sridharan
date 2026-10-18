#!/bin/bash
# Trial (3 steps) then arm A (100 steps), each on freshly started samplers (pushed weights persist in vLLM).
# Samplers: two independent DP1 servers (GPU 4 :8400, GPU 5 :8401); trainer: torchrun on GPUs 6-7.
cd /data/agastyas/rh_rl
KEY=$(cat /data/agastyas/rh_probe/api_key)
start_samplers() {
  tmux kill-session -t rl_vllm0 2>/dev/null; tmux kill-session -t rl_vllm1 2>/dev/null; sleep 8
  tmux new-session -d -s rl_vllm0 "VLLM_SERVER_DEV_MODE=1 GPUS=4 DP=1 PORT=8400 bash serve_gptoss_rl.sh > $1/vllm0.log 2>&1"
  tmux new-session -d -s rl_vllm1 "VLLM_SERVER_DEV_MODE=1 GPUS=5 DP=1 PORT=8401 bash serve_gptoss_rl.sh > $1/vllm1.log 2>&1"
  for p in 8400 8401; do
    for i in $(seq 1 90); do curl -s -o /dev/null -H "Authorization: Bearer $KEY" http://127.0.0.1:$p/v1/models && break; sleep 10; done
  done
  echo "samplers up $(date)"
}
train() {  # RUN STEPS extra-args...
  RUN=$1; STEPS=$2; shift 2
  rm -rf $RUN; mkdir -p $RUN
  start_samplers $RUN
  source env.sh
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/data/agastyas/rh_rl/tmp/triton \
  CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=8 venv/bin/torchrun --nproc_per_node 2 --master_port 29611 rl_train_gptoss.py \
    --rundir $RUN --steps $STEPS --mb-tokens 49152 "$@" > $RUN/train.log 2>&1
  echo "$RUN exit $? $(date); steps logged: $(wc -l < $RUN/log.jsonl 2>/dev/null)"
  [ "$(wc -l < $RUN/log.jsonl 2>/dev/null)" = "$STEPS" ]
}
# Arm B: queued after run_arms.sh (arm A) prints ALL DONE; aborts if arm A failed.
until grep -q "ALL DONE\|FAILED" logs/run_arms.log; do sleep 60; done
grep -q "FAILED" logs/run_arms.log && { echo "arm A failed; arm B not started $(date)"; exit 1; }
train runs/armB 100 --lam 1 --tau 770537 --save-every 10 || { echo "ARM B FAILED"; exit 1; }
echo "ARM B DONE $(date)"
