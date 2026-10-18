source /data/agastyas/rh_probe/env.sh
V=/data/agastyas/latekey/vllm-env
export CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13 PATH=$V/lib/python3.12/site-packages/nvidia/cu13/bin:$V/bin:$PATH
export HF_HUB_OFFLINE=1 HF_HUB_CACHE=/data/hf_cache VLLM_CACHE_ROOT=/data/agastyas/rh_probe/vllm_cache TMPDIR=/data/agastyas/rh_probe/tmp FLASHINFER_WORKSPACE_BASE=/data/agastyas/rh_probe/tmp
export CUDA_VISIBLE_DEVICES=${GPUS:-4,5,6,7}
exec vllm serve $(ls -d /data/hf_cache/models--zai-org--GLM-4.5-Air-FP8/snapshots/*) \
  --served-model-name glm-4.5-air --host 0.0.0.0 --port ${PORT:-8200} --api-key $(cat /data/agastyas/rh_probe/api_key) \
  --tensor-parallel-size ${TP:-2} --data-parallel-size ${DP:-2} --max-model-len 131072 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser glm45 --reasoning-parser glm45 \
  --disable-custom-all-reduce
