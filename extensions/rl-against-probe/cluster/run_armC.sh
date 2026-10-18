#!/bin/bash
# Arm C: continuous probe penalty. reward = pass - (score - tau) / 448548  (unclipped; NaN score -> 0).
# The scale makes the expected hack-vs-honest penalty gap at step 0 equal to arm B's (P(fire|hack) - P(fire|~hack) = 0.42;
# mean score gap 188,390 on A's steps 1-5, judged). Same seed / task order as A and B.
# Training: samplers GPUs 4,5 (ports 8400, 8401), trainer 2 ranks on GPUs 6,7. Then held-out eval on all 8 GPUs.
cd /data/agastyas/rh_rl
KEY=$(cat /data/agastyas/rh_probe/api_key)
source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/data/agastyas/rh_rl/tmp/triton OMP_NUM_THREADS=8
R=runs/armC; mkdir -p $R runs/eval

tmux new-session -d -s rl_vllm_C0 "VLLM_SERVER_DEV_MODE=1 GPUS=4 DP=1 PORT=8400 bash serve_gptoss_rl.sh > $R/vllm0.log 2>&1"
tmux new-session -d -s rl_vllm_C1 "VLLM_SERVER_DEV_MODE=1 GPUS=5 DP=1 PORT=8401 bash serve_gptoss_rl.sh > $R/vllm1.log 2>&1"
for p in 8400 8401; do
  for i in $(seq 1 90); do curl -s -o /dev/null -H "Authorization: Bearer $KEY" http://127.0.0.1:$p/v1/models && break; sleep 10; done
done
echo "armC samplers up $(date -u)"
CUDA_VISIBLE_DEVICES=6,7 venv/bin/torchrun --nproc_per_node 2 --master_port 29641 rl_train_gptoss.py --rundir $R \
  --ports 8400,8401 --lam 1 --tau 770537 --penalty linear --pen-scale 448548 \
  --steps 100 --prompts 16 --group 8 --mb-tokens 49152 --save-every 5 > $R/train.log 2>&1
rc=$?
echo "armC trainer exited $(date -u): rc=$rc, $(wc -l < $R/log.jsonl) steps logged"
touch $R/DONE
tmux kill-session -t rl_vllm_C0; tmux kill-session -t rl_vllm_C1
[ -d $R/ckpt_final_100 ] || { echo "no ckpt_final_100; skipping eval"; exit 1; }

lane() {  # lane name, sampler GPU, port, scorer GPU, master port, then jobs "name|mode:k"
  local L=$1 sg=$2 port=$3 tg=$4 mp=$5; shift 5
  for g in $sg $tg; do
    while [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 15; done
  done
  tmux new-session -d -s rl_evalvllm_$L "VLLM_SERVER_DEV_MODE=1 GPUS=$sg DP=1 PORT=$port bash serve_gptoss_rl.sh > runs/eval/vllm_$L.log 2>&1"
  for i in $(seq 1 90); do curl -s -o /dev/null -H "Authorization: Bearer $KEY" http://127.0.0.1:$port/v1/models && break; sleep 10; done
  for job in "$@"; do
    IFS='|' read -r name modes <<< "$job"
    echo "lane $L: eval $name ($modes) start $(date -u)"
    CUDA_VISIBLE_DEVICES=$tg venv/bin/torchrun --nproc_per_node 1 --master_port $mp rl_train_gptoss.py --rundir runs/eval/$name \
      --ports $port --resume $R/ckpt_final_100 --eval-out runs/eval/$name --eval-modes $modes --mb-tokens 49152 > runs/eval/$name.log 2>&1
    echo "lane $L: eval $name exit rc=$? $(date -u)"
  done
  tmux kill-session -t rl_evalvllm_$L
}
# lenient 8/task split 3+3+2 over three lanes, strict 4/task on the fourth: all four end at about the same time
lane C1 0 8410 1 29651 "armC_len|lenient:3" &
lane C2 2 8411 3 29652 "armC_len2|lenient:3" &
lane C3 4 8400 6 29653 "armC_len3|lenient:2" &
lane C4 5 8401 7 29654 "armC_str|strict:4" &
wait
touch runs/eval/DONE_C
echo "armC eval done $(date -u)"
