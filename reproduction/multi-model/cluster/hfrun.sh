#!/bin/bash
# usage: GPU=5 hfrun.sh <log-name> <script.py> [args...]  -- HF/torch jobs on athena02 with the fla/sklearn overlay
source /data/agastyas/rh_probe/env.sh
export HF_HUB_OFFLINE=1 PYTHONPATH=/data/agastyas/rh_probe/pylib CUDA_VISIBLE_DEVICES=${GPU:?set GPU} TMPDIR=/data/agastyas/rh_probe/tmp TRITON_CACHE_DIR=/data/agastyas/rh_probe/tmp/triton
N=$1; shift
cd /data/agastyas/rh_probe && exec $PY "$@" > logs/$N.log 2>&1
