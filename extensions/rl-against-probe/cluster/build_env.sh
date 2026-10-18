source /data/agastyas/rh_rl/env.sh
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$HOME/.cache/uv
uv venv --python 3.12 venv && uv pip install --python venv/bin/python -r base_freeze.txt && uv pip install --python venv/bin/python peft flash-linear-attention && venv/bin/python -c "import torch, vllm, transformers, peft, fla; print(\"ENV OK\", torch.__version__, vllm.__version__, transformers.__version__, peft.__version__, fla.__version__)"
