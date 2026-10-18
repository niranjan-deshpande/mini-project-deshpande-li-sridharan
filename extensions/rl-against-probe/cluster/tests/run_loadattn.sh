#!/bin/bash
cd /data/agastyas/rh_rl
tmux new-session -d -s rl_vllm0 "VLLM_SERVER_DEV_MODE=1 GPUS=4 DP=1 PORT=8400 bash serve_gptoss_rl.sh > logs/serve_loadattn.log 2>&1"
for i in $(seq 1 90); do curl -s -o /dev/null -H "Authorization: Bearer $(cat /data/agastyas/rh_probe/api_key)" http://127.0.0.1:8400/v1/models && break; sleep 10; done
venv/bin/python tests/test_loadattn.py > logs/t_loadattn.log 2>&1; echo "LA EXIT $?" >> logs/t_loadattn.log
