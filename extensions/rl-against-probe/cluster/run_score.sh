#!/bin/bash
# Cross-model probe re-scoring of the held-out lenient eval texts, all 36 layers x 21 DoM vectors (+ mean-pooled acts).
# Models: base (no adapter), A@100, B@100; texts: base / A / B eval samples (2472 each, in 1236-sample chunks).
# GPUs 0-3, one worker each; the base<->B cells (the core question) run first on every GPU.
cd /data/agastyas/rh_rl
source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/data/agastyas/rh_rl/tmp/triton OMP_NUM_THREADS=8
mkdir -p runs/score
BL_A=runs/eval/base_len/lenient.jsonl:0:1236; BL_B=runs/eval/base_len/lenient.jsonl:1236:2472
A_A=runs/eval/armA_len/lenient.jsonl:0:1236;  A_B=runs/eval/armA_len2/lenient.jsonl:0:1236
B_A=runs/eval/armB_len/lenient.jsonl:0:1236;  B_B=runs/eval/armB_len2/lenient.jsonl:0:1236
CK_A=runs/armA/ckpt_final_100; CK_B=runs/armB/ckpt_final_100
job() {  # gpu, master port, model tag, ckpt or -, comma list of chunks
  local res=""; [ "$4" != "-" ] && res="--resume $4"
  echo "gpu $1: $3 on $5 start $(date -u)"
  CUDA_VISIBLE_DEVICES=$1 venv/bin/torchrun --nproc_per_node 1 --master_port $2 rl_train_gptoss.py --rundir runs/score/$3 $res \
    --score-in $5 --score-out runs/score --mb-tokens 49152 >> runs/score/worker_$1.log 2>&1
  echo "gpu $1: $3 exit rc=$? $(date -u)"
}
( job 0 29671 base - $BL_A,$BL_B; job 0 29671 base - $A_A,$A_B; job 0 29671 A $CK_A $B_A ) &
( job 1 29672 B $CK_B $BL_A,$BL_B; job 1 29672 B $CK_B $A_A,$A_B; job 1 29672 A $CK_A $B_B ) &
( job 2 29673 base - $B_A,$B_B;    job 2 29673 A $CK_A $A_A,$A_B ) &
( job 3 29674 B $CK_B $B_A,$B_B;   job 3 29674 A $CK_A $BL_A,$BL_B ) &
wait
touch runs/score/DONE
echo "scoring done $(date -u)"
