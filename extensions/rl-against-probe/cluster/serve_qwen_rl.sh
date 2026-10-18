# Qwen3.8-27B for RL sampling: one TP1 engine per GPU via data parallel. LORA=1 adds LoRA serving.
source /data/agastyas/rh_rl/env.sh
V=/data/agastyas/rh_rl/venv
export CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13 PATH=$V/lib/python3.12/site-packages/nvidia/cu13/bin:$V/bin:$PATH
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=/data/hf_cache VLLM_CACHE_ROOT=/data/agastyas/rh_rl/vllm_cache FLASHINFER_WORKSPACE_BASE=/data/agastyas/rh_rl/tmp
export CUDA_VISIBLE_DEVICES=${GPUS:-4,5,6,7}
LORA_ARGS=""; [ "$LORA" = "1" ] && LORA_ARGS="--enable-lora --max-lora-rank 64 --max-loras 2"
exec vllm serve $(ls -d /data/hf_cache/models--Qwen--Qwen3.8-27B/snapshots/*) \
  --served-model-name qwen3.8-27b --host 127.0.0.1 --port ${PORT:-8400} --api-key $(cat /data/agastyas/rh_probe/api_key) \
  --data-parallel-size ${DP:-4} --max-model-len 24576 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image":0,"video":0}' --enable-prefix-caching --disable-custom-all-reduce $LORA_ARGS
