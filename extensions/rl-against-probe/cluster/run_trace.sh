#!/bin/bash
# Per-token layer-13 span__all traces (+ decoded peak windows) for base/B weights x base/B held-out lenient texts.
# Same re-encoding as run_score.sh; outputs runs/score_trace/{model}__{chunk}_trace13.npz and _peaks13.jsonl. GPUs 4-7.
cd /data/agastyas/rh_rl
source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/data/agastyas/rh_rl/tmp/triton OMP_NUM_THREADS=8
mkdir -p runs/score_trace
BL_A=runs/eval/base_len/lenient.jsonl:0:1236; BL_B=runs/eval/base_len/lenient.jsonl:1236:2472
B_A=runs/eval/armB_len/lenient.jsonl:0:1236;  B_B=runs/eval/armB_len2/lenient.jsonl:0:1236
CK_B=runs/armB/ckpt_final_100
job() {  # gpu, master port, model tag, ckpt or -, comma list of chunks
  local res=""; [ "$4" != "-" ] && res="--resume $4"
  echo "gpu $1: $3 on $5 start $(date -u)"
  CUDA_VISIBLE_DEVICES=$1 venv/bin/torchrun --nproc_per_node 1 --master_port $2 rl_train_gptoss.py --rundir runs/score_trace/$3 $res \
    --score-in $5 --score-out runs/score_trace --score-trace 13 --mb-tokens 49152 >> runs/score_trace/worker_$1.log 2>&1
  echo "gpu $1: $3 exit rc=$? $(date -u)"
}
( job 4 29681 base - $BL_A; job 4 29681 base - $B_A ) &
( job 5 29682 base - $BL_B; job 5 29682 base - $B_B ) &
( job 6 29683 B $CK_B $B_A; job 6 29683 B $CK_B $BL_A ) &
( job 7 29684 B $CK_B $B_B; job 7 29684 B $CK_B $BL_B ) &
wait
touch runs/score_trace/DONE
echo "trace done $(date -u)"
