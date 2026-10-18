source /data/agastyas/rh_probe/env.sh
export HF_HUB_OFFLINE=0 HF_HOME=/data/agastyas/rh_probe/hf_home PYTHONPATH=/data/agastyas/rh_probe/pylib CUDA_VISIBLE_DEVICES=5 TMPDIR=/data/agastyas/rh_probe/tmp TRITON_CACHE_DIR=/data/agastyas/rh_probe/tmp/triton
cd /data/agastyas/rh_probe && exec $PY build_dom.py --model $(ls -d /data/agastyas/hf_cache_latekey/models--openai--gpt-oss-120b/snapshots/*) --family gptoss --out dom/gptoss > logs/t1_gptoss.log 2>&1
