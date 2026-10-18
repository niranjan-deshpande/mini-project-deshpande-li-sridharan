#!/bin/bash
# Runs on the GPU box. Installs deps, runs the J-lens experiment for both OLMo models, writes JSON results.
# Self-terminate safety: the instance is launched with shutdown-behavior=terminate, and we schedule a hard
# failsafe shutdown at +150 min up front so a hang can never bill forever. On clean success we shut down soon;
# on failure we LEAVE THE BOX UP (failsafe still armed) so the run can be debugged.
set -uo pipefail
sudo shutdown -h +70 || true           # hard failsafe (terminate) ~70min from boot (instruct-only re-run)
export DEBIAN_FRONTEND=noninteractive
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=$HOME/hf
cd "$HOME/jlens"

echo "=== $(date) setup ==="
# base AMI has no python3-venv; use system python3 with --user installs (lands in ~/.local)
export PATH=$HOME/.local/bin:$PATH
rm -f DONE FAILED
python3 -m pip install --user -q -U pip wheel
python3 -m pip install --user -q "torch" "transformers==5.19.0" accelerate huggingface_hub hf_transfer numpy
echo "=== deps ==="
python3 -c "import torch,transformers; print('torch',torch.__version__,'tf',transformers.__version__,'cuda',torch.cuda.is_available(),'ngpu',torch.cuda.device_count())"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
if ! python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "=== NO CUDA — aborting (box stays up, failsafe armed) ==="; touch FAILED; exit 1
fi

# shared borrowed lens (cached under HF_HOME, kept across both models)
python3 - <<'PY'
from huggingface_hub import hf_hub_download
print("jpp lens ->", hf_hub_download("koayon/jpp-lenses", "olmo-3-1125-32b/lens.pt"))
PY

run_one () {
  local TAG=$1 MODEL=$2 DOM=$3 EXIST=$4
  echo "=== $(date) run $TAG ($MODEL) ==="
  python3 jlens_run.py --model "$MODEL" --dom "$DOM" --existing-logit "$EXIST" \
      --pairs pairs_full.jsonl --jpp koayon/jpp-lenses --out "jlens_${TAG}.json" --tag "$TAG"
  local rc=$?
  # free the model weights from cache before the next one
  rm -rf "$HF_HOME"/hub/models--allenai--* 2>/dev/null || true
  return $rc
}

ok=1
# instruct-only re-run (Think already completed on the prior box)
run_one instruct allenai/Olmo-3.1-32B-Instruct dom_instruct.npz logit_instruct_existing.json || ok=0

if [ "$ok" = 1 ] && [ -f jlens_instruct.json ]; then
  touch DONE
  echo "=== $(date) INSTRUCT DONE, shutting down (terminate) in 3 min ==="
  sudo shutdown -h +3 || true
else
  touch FAILED
  echo "=== $(date) FAILED — leaving box up for debug (failsafe +70 still armed) ==="
fi
