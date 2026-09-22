#!/bin/bash
# r8 head-only retrain with OOM retry: batch 4 -> 2 -> 1 (accum scaled so the effective batch stays 16).
REPO=$(cd "$(dirname "$0")/.." && pwd); PY=/opt/conda/envs/shared_audio/bin/python; cd "$REPO"
GPU=$1; RUN=posterior_r8_warm_headonly
for BS in 4 2 1; do
  ACC=$((16 / BS))
  rm -rf outputs/posterior/$RUN
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY src/train_posterior.py --mode r8 --run $RUN --gpu 0 --init warm --tau 0.3 --head-only --head-lr 1e-3 --epochs 30 --patience 6 --batch-size $BS --accum $ACC --num-workers 4 > logs/train_${RUN}_bs$BS.log 2>&1
  if [ -f outputs/posterior/$RUN/train_done.json ]; then echo "finished with batch $BS" > logs/train_${RUN}.status; exit 0; fi
  if grep -q "OutOfMemoryError" logs/train_${RUN}_bs$BS.log; then echo "OOM at batch $BS, retrying" >> logs/train_${RUN}.status; continue; fi
  echo "failed (not OOM) at batch $BS" >> logs/train_${RUN}.status; exit 1
done
echo "OOM at every batch size" >> logs/train_${RUN}.status; exit 1
