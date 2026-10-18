#!/bin/bash
# Resume both arms from ckpt_10 after the 04:34 crash (20k-token prompt + 16k max_tokens > 32k context -> HTTP 400).
cd /data/agastyas/rh_rl
STOP_AT=1791960300   # 2026-10-14 06:45:00 UTC
KEY=$(cat /data/agastyas/rh_probe/api_key)
for r in armA armB; do
  head -n 10 runs/$r/log.jsonl > runs/$r/log.tmp && mv runs/$r/log.tmp runs/$r/log.jsonl
  mkdir -p runs/$r/samples_superseded; mv runs/$r/samples/step_001[01].jsonl runs/$r/samples_superseded/ 2>/dev/null
  mv runs/$r/train.log runs/$r/train_crash0434.log; mv runs/$r/vllm.log runs/$r/vllm_crash0434.log
done
tmux new-session -d -s rl_vllm0 "VLLM_SERVER_DEV_MODE=1 GPUS=4 DP=1 PORT=8400 bash serve_gptoss_rl.sh > runs/armA/vllm.log 2>&1"
tmux new-session -d -s rl_vllm1 "VLLM_SERVER_DEV_MODE=1 GPUS=5 DP=1 PORT=8401 bash serve_gptoss_rl.sh > runs/armB/vllm.log 2>&1"
for p in 8400 8401; do
  for i in $(seq 1 90); do curl -s -o /dev/null -H "Authorization: Bearer $KEY" http://127.0.0.1:$p/v1/models && break; sleep 10; done
done
echo "samplers up $(date)"
source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/data/agastyas/rh_rl/tmp/triton OMP_NUM_THREADS=8
COMMON="--steps 60 --prompts 16 --group 8 --mb-tokens 49152 --save-every 5 --stop-at $STOP_AT"
CUDA_VISIBLE_DEVICES=6 venv/bin/torchrun --nproc_per_node 1 --master_port 29611 rl_train_gptoss.py --rundir runs/armA --ports 8400 --lam 0 --resume runs/armA/ckpt_10 $COMMON > runs/armA/train.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 venv/bin/torchrun --nproc_per_node 1 --master_port 29612 rl_train_gptoss.py --rundir runs/armB --ports 8401 --lam 1 --tau 770537 --resume runs/armB/ckpt_10 $COMMON > runs/armB/train.log 2>&1 &
wait
echo "trainers exited $(date): A $(wc -l < runs/armA/log.jsonl) steps, B $(wc -l < runs/armB/log.jsonl) steps"
touch runs/armA/DONE runs/armB/DONE
tmux kill-session -t rl_vllm0; tmux kill-session -t rl_vllm1
echo "samplers stopped $(date)"
