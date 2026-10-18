source /data/agastyas/rh_probe/env.sh
V=/data/agastyas/latekey/vllm-env
export CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13 PATH=$V/lib/python3.12/site-packages/nvidia/cu13/bin:$V/bin:$PATH
export HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=/data/agastyas/rh_probe/vllm_cache TMPDIR=/data/agastyas/rh_probe/tmp FLASHINFER_WORKSPACE_BASE=/data/agastyas/rh_probe/tmp
export CUDA_VISIBLE_DEVICES=${GPUS:-6,7}
exec vllm serve /data/agastyas/hf_cache_latekey/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --served-model-name gpt-oss-120b --host 0.0.0.0 --port 8200 --api-key $(cat /data/agastyas/rh_probe/api_key) \
  --data-parallel-size ${DP:-2} --max-model-len 131072 --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss --disable-custom-all-reduce
