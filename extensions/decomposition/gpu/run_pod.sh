#!/bin/bash
# Runs ON the pod, unattended. Sets up, downloads weights to the network volume, runs session1, then deletes the pod
# whether the job succeeded or failed (outputs stay on the network volume at /workspace/out).
#   nohup bash /workspace/code/gpu/run_pod.sh > /workspace/run_pod.log 2>&1 &
set -u
W=/workspace
STAGES=${STAGES:-check,pairs,dirs,rollouts,lenses,manifest}
POD_ID=${POD_ID:?set POD_ID}
terminate() {
  rc=$?; set +x   # never echo the API key
  if [ "$rc" != "0" ] || [ -f $W/out/FAILED ] || [ -f $W/out/emotions/FAILED ]; then
    echo "[$(date +%T)] job failed (rc $rc); waiting 20 min before deleting (touch $W/HOLD to keep the pod)"
    for i in $(seq 1 40); do sleep 30; [ -f $W/HOLD ] && { echo "HOLD found; not deleting"; return; }; done
  fi
  echo "[$(date +%T)] deleting pod $POD_ID"; sync
  KEY=$(grep -i '^apikey' /root/.runpod/config.toml | head -1 | sed -E "s/^[^=]*= *//; s/^[\"']//; s/[\"']$//")
  for i in 1 2 3 4 5; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "https://rest.runpod.io/v1/pods/$POD_ID" -H "Authorization: Bearer $KEY")
    echo "delete -> $code"; [ "$code" = "200" ] || [ "$code" = "204" ] && break; sleep 20
  done
}
trap terminate EXIT
set -x
cd $W/code
python3 -m venv --system-site-packages $W/venv
$W/venv/bin/pip install -q "transformers==5.19.0" accelerate "flash-linear-attention==0.5.2" huggingface_hub numpy pyarrow scikit-learn scipy
$W/venv/bin/python -c "import torch, fla, transformers; print(torch.__version__, torch.cuda.get_device_name(0), transformers.__version__)" || exit 1
[ -f $W/models/Qwen3.8-27B/config.json ] || $W/venv/bin/hf download Qwen/Qwen3.8-27B --local-dir $W/models/Qwen3.8-27B || exit 1
if [ "${SESSION:-1}" = "1" ] && [ ! -f $W/models/jpp/qwen3.6-27b/lens.pt ]; then
  $W/venv/bin/hf download koayon/jpp-lenses qwen3.6-27b/lens.pt --local-dir $W/models/jpp || echo "J++ lens download failed; skipping borrowed J++"
fi
export JPP_LENS=$W/models/jpp/qwen3.6-27b/lens.pt
if [ "${SESSION:-1}" = "2" ]; then
  timeout 2h $W/venv/bin/python gpu/session2_emotions.py --model $W/models/Qwen3.8-27B --data $W/data --out $W/out
else
  timeout 5h $W/venv/bin/python gpu/session1.py --model $W/models/Qwen3.8-27B --data $W/data --out $W/out --stages "$STAGES"
fi
rc=$?; echo "[$(date +%T)] session exit code $rc"; exit $rc
