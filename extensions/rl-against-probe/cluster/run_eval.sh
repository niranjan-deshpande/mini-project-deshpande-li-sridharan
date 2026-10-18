#!/bin/bash
# Held-out eval on the 309 ImpossibleBench-LCB tasks (103 x original/oneoff/conflicting), all 8 GPUs, 4 lanes.
# One (model, prompt mode) per job; each lane = 1 sampler GPU + 1 scorer GPU, and each job first pushes its own merged
# attention weights to the lane's sampler (base = identity LoRA), so a lane can switch models without a restart.
#   lane B  (GPUs 5/7): B lenient (half) once armB is DONE
#   lane 01 (GPUs 0/1): B lenient (other half), then A strict once armA is DONE
#   lane 23 (GPUs 2/3): B strict, then A lenient (other half)
#   lane A  (GPUs 4/6): A lenient (half) once armA is DONE
cd /data/agastyas/rh_rl
KEY=$(cat /data/agastyas/rh_probe/api_key)
source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/data/agastyas/rh_rl/tmp/triton OMP_NUM_THREADS=8
mkdir -p runs/eval

lane() {  # lane name, sampler GPU, port, scorer GPU, master port, then jobs "name|ckpt or -|mode:k|arm to wait for or -"
  local L=$1 sg=$2 port=$3 tg=$4 mp=$5; shift 5
  local up=0
  for job in "$@"; do
    IFS='|' read -r name ck modes waitarm <<< "$job"
    if [ "$waitarm" != "-" ]; then
      while [ ! -f runs/$waitarm/DONE ]; do sleep 30; done
    fi
    if [ $up = 0 ]; then
      for g in $sg $tg; do  # training sampler / trainer must have released the GPU
        while [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 15; done
      done
      tmux new-session -d -s rl_evalvllm_$L "VLLM_SERVER_DEV_MODE=1 GPUS=$sg DP=1 PORT=$port bash serve_gptoss_rl.sh > runs/eval/vllm_$L.log 2>&1"
      for i in $(seq 1 90); do curl -s -o /dev/null -H "Authorization: Bearer $KEY" http://127.0.0.1:$port/v1/models && break; sleep 10; done
      echo "lane $L: sampler up on GPU $sg $(date -u)"; up=1
    fi
    local res=""; [ "$ck" != "-" ] && res="--resume $ck"
    echo "lane $L: eval $name ($ck, $modes) start $(date -u)"
    CUDA_VISIBLE_DEVICES=$tg venv/bin/torchrun --nproc_per_node 1 --master_port $mp rl_train_gptoss.py --rundir runs/eval/$name \
      --ports $port $res --eval-out runs/eval/$name --eval-modes $modes --mb-tokens 49152 > runs/eval/$name.log 2>&1
    local rc=$?
    echo "lane $L: eval $name exit rc=$rc $(date -u); $(grep -c '"eval"' runs/eval/$name.log) modes written"
  done
  tmux kill-session -t rl_evalvllm_$L
  echo "lane $L done $(date -u)"
}
# (base_len / base_str finished 18:34 / 17:51 in the first version of this chain.) Each arm's 8 lenient samples per task
# are split into two 4-sample jobs on different lanes so that all six jobs end at about the same time.
lane B  5 8401 7 29622 "armB_len|runs/armB/ckpt_final_100|lenient:4|armB" &
lane 01 0 8410 1 29631 "armB_len2|runs/armB/ckpt_final_100|lenient:4|armB" "armA_str|runs/armA/ckpt_final_100|strict:4|armA" &
lane 23 2 8411 3 29632 "armB_str|runs/armB/ckpt_final_100|strict:4|armB"  "armA_len2|runs/armA/ckpt_final_100|lenient:4|armA" &
lane A  4 8400 6 29621 "armA_len|runs/armA/ckpt_final_100|lenient:4|armA" &
wait
touch runs/eval/DONE
echo "all evals done $(date -u)"
