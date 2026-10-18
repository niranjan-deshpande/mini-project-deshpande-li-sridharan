# second round: stop the spent servers (cond 0 done; neg3 stopped early, model breaks), run -1x and -1.5x on GPUs 4 and 6
cd /data/agastyas/rh_probe
PY=/data/agastyas/latekey/vllm-env/bin/python
kill $(cat steer_neg/serve_neg3.pid) $(cat steer_neg/serve_0.pid)
$PY -c "
import numpy as np; v=np.load('steer_neg/neg2.npy')/-2
for k in (1, 1.5): np.save(f'steer_neg/neg{k}.npy', -k*v); print(k, np.linalg.norm(-k*v))"
until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4) -lt 1000 ] && [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6) -lt 1000 ]; do sleep 5; done
echo "gpus free $(date)"
GPU=4 PORT=8300 COND=neg1.5 nohup bash steer_neg/serve.sh > steer_neg/serve_neg1.5.log 2>&1 & echo $! > steer_neg/serve_neg1.5.pid
GPU=6 PORT=8302 COND=neg1 nohup bash steer_neg/serve.sh > steer_neg/serve_neg1.log 2>&1 & echo $! > steer_neg/serve_neg1.pid
for p in 8300 8302; do until curl -sf localhost:$p/health >/dev/null; do sleep 15
  for c in neg1 neg1.5; do kill -0 $(cat steer_neg/serve_$c.pid) 2>/dev/null || { echo "server $c died" > steer_neg/FAILED2; exit 1; }; done; done; done
for c in neg1 neg1.5; do grep -q "\[steer\] layer 33" steer_neg/serve_$c.log || { echo "no steer line $c" > steer_neg/FAILED2; exit 1; }; done
echo "servers up $(date)"
$PY steer_neg/client.py 8300 neg1.5 2 > steer_neg/client_neg1.5.log 2>&1 & A=$!
$PY steer_neg/client.py 8302 neg1 2 > steer_neg/client_neg1.log 2>&1 & B=$!
wait $A $B
kill $(cat steer_neg/serve_neg1.pid) $(cat steer_neg/serve_neg1.5.pid)
echo "ALL DONE $(date)" > steer_neg/DONE2
