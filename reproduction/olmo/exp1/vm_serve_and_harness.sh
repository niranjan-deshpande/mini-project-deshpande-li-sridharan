#!/bin/bash
# Run on the GPU VM after vm_setup.sh completes.
# 1) installs the patched ImpossibleBench; 2) starts vLLM in tmux.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source ~/venvs/vllm/bin/activate

# --- patched harness ---
if [ ! -d ~/impossiblebench ]; then
  git clone https://github.com/safety-research/impossiblebench ~/impossiblebench
  cd ~/impossiblebench
  git checkout 061dc3d
  git apply ~/impossiblebench-lcb-fixes.patch
  uv pip install -e .
fi

# --- docker access for inspect sandboxes ---
sudo usermod -aG docker "$USER" || true

# --- vLLM server in tmux ---
tmux kill-session -t vllm 2>/dev/null || true
tmux new-session -d -s vllm "source ~/venvs/vllm/bin/activate && \
  vllm serve ~/models/olmo-3.1-32b-think \
    --served-model-name olmo \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.92 \
    --port 8000 2>&1 | tee ~/vllm.log"
echo "vLLM starting in tmux session 'vllm' (log: ~/vllm.log)"
