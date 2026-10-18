#!/bin/bash
cd /data/agastyas/rh_rl
for i in $(seq 1 90); do curl -s -o /dev/null -H "Authorization: Bearer $(cat /data/agastyas/rh_probe/api_key)" http://127.0.0.1:8400/v1/models && break; sleep 10; done
echo "sampler up $(date)"
source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/data/agastyas/rh_rl/tmp/triton CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=8
rm -rf runs/trialA
venv/bin/torchrun --nproc_per_node 2 --master_port 29611 rl_train_gptoss.py --rundir runs/trialA --lam 0 --steps 3 --mb-tokens 49152 --save-every 1000 > logs/trialA.log 2>&1
echo "TRIAL EXIT $? $(date)"
