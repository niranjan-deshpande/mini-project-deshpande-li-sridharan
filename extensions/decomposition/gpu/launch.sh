#!/bin/bash
# Run LOCALLY (needs Niranjan's go-ahead): create volume + pod, upload code/data, start the unattended job.
# Usage: bash gpu/launch.sh [GPU_ID] [DC]
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${1:-"NVIDIA H100 80GB HBM3"}; DC=${2:-AP-JP-1}
IMG=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
VOL=$(runpodctl network-volume list | python3 -c "import json,sys; v=[x for x in json.load(sys.stdin) if x['name']=='rh-decomp']; print(v[0]['id'] if v else '')")
if [ -z "$VOL" ]; then
  VOL=$(runpodctl network-volume create --name rh-decomp --size 200 --data-center-id "$DC" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
fi
echo "volume $VOL"
# package code + data (small: final pairs, rollouts, Agastya's vectors/scores, pairs_full)
rm -rf /tmp/rhpkg && mkdir -p /tmp/rhpkg/data/agastya /tmp/rhpkg/code
cp -r data/final /tmp/rhpkg/data/
cp ../../reproduction/olmo/synth/pairs_full.jsonl ../../reproduction/multi-model/results/rollouts/qwen_rollouts.jsonl.gz /tmp/rhpkg/data/
cp ../../reproduction/multi-model/results/dom/qwen/dom_vectors.npz /tmp/rhpkg/data/agastya/
cp -r ../../reproduction/multi-model/results/t2 /tmp/rhpkg/data/agastya/
rsync -a --exclude .venv --exclude data --exclude __pycache__ --exclude .git ./ /tmp/rhpkg/code/
tar -C /tmp/rhpkg -czf /tmp/rhpkg.tgz code data
ls -la /tmp/rhpkg.tgz
HF=$(grep '^HF_TOKEN=' ../../.env | cut -d= -f2-)
POD=$(runpodctl pod create --name rh-decomp-s1 --image "$IMG" --gpu-id "$GPU" --data-center-ids "$DC" \
  --network-volume-id "$VOL" --container-disk-in-gb 40 --ports "22/tcp" --env "{\"HF_TOKEN\":\"$HF\"}" --wait \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['id'])")
echo "pod $POD"; runpodctl pod get "$POD"
