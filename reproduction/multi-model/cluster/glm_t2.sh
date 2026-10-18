#!/bin/bash
# usage: GPUS=6,7 SHARD=0/2 bash glm_t2.sh
source /data/agastyas/rh_probe/env.sh
export HF_HUB_OFFLINE=1 PYTHONPATH=/data/agastyas/rh_probe/pylib CUDA_VISIBLE_DEVICES=${GPUS:?} TMPDIR=/data/agastyas/rh_probe/tmp TRITON_CACHE_DIR=/data/agastyas/rh_probe/tmp/triton PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=$(ls -d /data/agastyas/hf_cache_latekey/models--zai-org--GLM-4.5-Air/snapshots/*)
cd /data/agastyas/rh_probe
exec $PY t2_acts_glm.py --model $M --dom dom/glm/dom_vectors.npz --rollouts out_glm_imp.jsonl --out t2/glm --shard ${SHARD:?} > logs/t2_glm_${SHARD/\//of}.log 2>&1
