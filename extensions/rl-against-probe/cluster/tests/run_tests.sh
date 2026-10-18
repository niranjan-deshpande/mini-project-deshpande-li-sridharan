#!/bin/bash
# T2 trainer smoke on GPU 5 (background) + T3 merged-weight reload against the dev-mode server on GPU 4.
cd /data/agastyas/rh_rl
source env.sh
(CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/data/agastyas/rh_rl/tmp/triton venv/bin/python -I tests/trainer_smoke.py > logs/t_trainer_smoke.log 2>&1; echo "SMOKE EXIT $?" >> logs/t_trainer_smoke.log) &
for i in $(seq 1 60); do curl -s -o /dev/null -H "Authorization: Bearer $(cat /data/agastyas/rh_probe/api_key)" http://127.0.0.1:8400/v1/models && break; sleep 10; done
venv/bin/python tests/test_reload.py > logs/t_reload.log 2>&1; echo "RELOAD EXIT $?" >> logs/t_reload.log
wait; echo "ALL DONE $(date)"
