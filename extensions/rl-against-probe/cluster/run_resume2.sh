#!/bin/bash
# Continue both arms to step 100 from the 06:44 deadline stop (A: ckpt_final_37, B: ckpt_final_41). No deadline.
# Each arm has its own lifecycle (sampler -> trainer -> DONE -> sampler killed), so B's GPUs (5, 7) free up as soon as
# B finishes and the held-out eval can start there while A is still training.
cd /data/agastyas/rh_rl
KEY=$(cat /data/agastyas/rh_probe/api_key)
for r in armA armB; do
  for f in STOPPED DONE; do [ -f runs/$r/$f ] && mv runs/$r/$f runs/$r/${f}_0644; done
  mv runs/$r/train.log runs/$r/train_0644.log; mv runs/$r/vllm.log runs/$r/vllm_0644.log
done
source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/data/agastyas/rh_rl/tmp/triton OMP_NUM_THREADS=8
COMMON="--steps 100 --prompts 16 --group 8 --mb-tokens 49152 --save-every 5"

arm() {  # arm name, sampler GPU, port, trainer GPU, master port, extra trainer args
  local r=$1 sg=$2 port=$3 tg=$4 mp=$5; shift 5
  tmux new-session -d -s rl_vllm_$r "VLLM_SERVER_DEV_MODE=1 GPUS=$sg DP=1 PORT=$port bash serve_gptoss_rl.sh > runs/$r/vllm.log 2>&1"
  for i in $(seq 1 90); do curl -s -o /dev/null -H "Authorization: Bearer $KEY" http://127.0.0.1:$port/v1/models && break; sleep 10; done
  echo "$r sampler up $(date -u)"
  local ck=$(ls -d runs/$r/ckpt_final_* | sort -t_ -k3 -n | tail -n 1)
  CUDA_VISIBLE_DEVICES=$tg venv/bin/torchrun --nproc_per_node 1 --master_port $mp rl_train_gptoss.py --rundir runs/$r \
    --ports $port --resume $ck "$@" $COMMON > runs/$r/train.log 2>&1
  local rc=$?
  echo "$r trainer exited $(date -u): rc=$rc, $(wc -l < runs/$r/log.jsonl) steps logged"
  touch runs/$r/DONE
  tmux kill-session -t rl_vllm_$r
  echo "$r sampler stopped $(date -u)"
}
arm armA 4 8400 6 29611 --lam 0 &
arm armB 5 8401 7 29612 --lam 1 --tau 770537 &
wait
echo "both arms done $(date -u)"
