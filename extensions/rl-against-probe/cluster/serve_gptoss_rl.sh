# gpt-oss-120b (MXFP4) for RL sampling: one TP1 engine per GPU via data parallel. LORA=1 adds LoRA serving.
source /data/agastyas/rh_rl/env.sh
V=/data/agastyas/rh_rl/venv
export CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13 PATH=$V/lib/python3.12/site-packages/nvidia/cu13/bin:$V/bin:$PATH
export HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=/data/agastyas/rh_rl/vllm_cache FLASHINFER_WORKSPACE_BASE=/data/agastyas/rh_rl/tmp
export CUDA_VISIBLE_DEVICES=${GPUS:-4,5,6,7} PYTHONPATH=/data/agastyas/rh_rl
LORA_ARGS=""; [ "$LORA" = "1" ] && LORA_ARGS="--enable-lora --max-lora-rank 64 --max-loras 2"
exec vllm serve /data/agastyas/hf_cache_latekey/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --served-model-name gpt-oss-120b --host 127.0.0.1 --port ${PORT:-8400} --api-key $(cat /data/agastyas/rh_probe/api_key) \
  --data-parallel-size ${DP:-4} --max-model-len 32768 --gpu-memory-utilization 0.92 --enable-prefix-caching \
  --reasoning-parser openai_gptoss --disable-custom-all-reduce --api-server-count 1 --worker-extension-cls vllm_ext.AttnLoader $LORA_ARGS
