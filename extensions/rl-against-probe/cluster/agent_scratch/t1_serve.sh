#!/bin/bash
# GPU TEST 1 server (UNTESTED; run only on GPUs agreed with the user). LoRA-enabled gpt-oss-120b, DP over $GPUS,
# single API server (required for runtime LoRA loading), sleep mode on. Usage: GPUS=4,5 DP=2 bash t1_serve.sh
source /data/agastyas/rh_rl/env.sh
V=/data/agastyas/rh_rl/venv
export CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13 PATH=$V/lib/python3.12/site-packages/nvidia/cu13/bin:$V/bin:$PATH
export HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=/data/agastyas/rh_rl/vllm_cache FLASHINFER_WORKSPACE_BASE=/data/agastyas/rh_rl/tmp
export CUDA_VISIBLE_DEVICES=$GPUS VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 VLLM_SERVER_DEV_MODE=1
exec vllm serve /data/agastyas/hf_cache_latekey/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --served-model-name gpt-oss-120b --host 127.0.0.1 --port ${PORT:-8410} --api-key $(cat /data/agastyas/rh_probe/api_key) \
  --data-parallel-size ${DP:-2} --api-server-count 1 --max-model-len 32768 --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --reasoning-parser openai_gptoss --disable-custom-all-reduce --enable-lora --max-lora-rank 32 --max-loras 2 \
  --enable-sleep-mode
