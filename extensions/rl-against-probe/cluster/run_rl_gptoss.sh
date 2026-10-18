#!/bin/bash
# One RL arm: vLLM sampler DP2 on GPUs 4-5 (no LoRA, dev mode for weight reloads) + 2-rank trainer on GPUs 6-7.
# Usage: ./run_rl_gptoss.sh RUN_DIR [trainer args...]     e.g. ./run_rl_gptoss.sh runs/armA --lam 0
cd /data/agastyas/rh_rl
RUN=$1; shift
mkdir -p $RUN
tmux kill-session -t rl_vllm 2>/dev/null; sleep 5
tmux new-session -d -s rl_vllm "VLLM_SERVER_DEV_MODE=1 GPUS=4,5 DP=2 bash serve_gptoss_rl.sh > $RUN/vllm.log 2>&1"
for i in $(seq 1 90); do curl -s -o /dev/null -H "Authorization: Bearer $(cat /data/agastyas/rh_probe/api_key)" http://127.0.0.1:8400/v1/models && break; sleep 10; done
echo "sampler up $(date)"
source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/data/agastyas/rh_rl/tmp/triton CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=8
venv/bin/torchrun --nproc_per_node 2 --master_port 29611 rl_train_gptoss.py --rundir $RUN "$@" > $RUN/train.log 2>&1
echo "TRAIN EXIT $? $(date)"
