#!/bin/bash
# usage: GPUS=4,5 bash glm_phase1.sh   -- T1 DoM vectors + logit lens, then T3 strength calibration (n=8)
source /data/agastyas/rh_probe/env.sh
export HF_HUB_OFFLINE=1 PYTHONPATH=/data/agastyas/rh_probe/pylib CUDA_VISIBLE_DEVICES=${GPUS:?} TMPDIR=/data/agastyas/rh_probe/tmp TRITON_CACHE_DIR=/data/agastyas/rh_probe/tmp/triton
M=$(ls -d /data/agastyas/hf_cache_latekey/models--zai-org--GLM-4.5-Air/snapshots/*)
cd /data/agastyas/rh_probe
$PY build_dom.py --model $M --family glm --out dom/glm > logs/t1_glm.log 2>&1 || { echo T1 FAILED; exit 1; }
$PY t3_steer.py --family glm --model $M --dom dom/glm/dom_vectors.npz --layer ${LAYER:-33} --out t3/glm_cal --conds 0,1,2,3,4,-3 --limit 8 --max-new 1024 > logs/t3_glm_cal.log 2>&1 || { echo T3CAL FAILED; exit 1; }
echo PHASE1 DONE
