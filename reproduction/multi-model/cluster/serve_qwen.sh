source /data/agastyas/rh_probe/env.sh
V=/data/agastyas/latekey/vllm-env
export CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13 PATH=$V/lib/python3.12/site-packages/nvidia/cu13/bin:$V/bin:$PATH
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=/data/hf_cache VLLM_CACHE_ROOT=/data/agastyas/rh_probe/vllm_cache TMPDIR=/data/agastyas/rh_probe/tmp FLASHINFER_WORKSPACE_BASE=/data/agastyas/rh_probe/tmp
export CUDA_VISIBLE_DEVICES=${GPUS:-2,3,4}
exec vllm serve /data/hf_cache/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --served-model-name qwen3.8-27b --host 0.0.0.0 --port ${PORT:-8100} --api-key $(cat /data/agastyas/rh_probe/api_key) \
  --data-parallel-size ${DP:-3} --max-model-len 131072 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt "{\"image\":0,\"video\":0}" --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --disable-custom-all-reduce
