#!/bin/bash
# E106b features: train-scene features for the flat heads (after their train predictions), and
# test features for the warm / head-only heads (after their test predictions). Liveness-checked.
REPO=$(cd "$(dirname "$0")/.." && pwd); PY=/opt/conda/envs/shared_audio/bin/python; cd "$REPO"
waitfor() { # $1 = done file, $2 = process pattern
  while [ ! -f "$1" ]; do if ! pgrep -f "$2" > /dev/null; then echo "process '$2' gone before $1" >> logs/e106b_chain.failed; exit 1; fi; sleep 60; done; }
waitfor logs/post_pred_r2_train.done posterior_predict.py
waitfor logs/post_pred_r8_train.done posterior_predict.py
$PY src/voxel_features.py --mode r2 --split train --workers 24 > logs/e106b_feat_r2_train.log 2>&1
$PY src/voxel_features.py --mode r8 --split train --workers 24 > logs/e106b_feat_r8_train.log 2>&1
waitfor logs/post_pred_heads_test.done posterior_predict.py
for h in warm warm_headonly; do for m in r2 r8; do
  $PY src/voxel_features.py --mode ${m}_${h} --split test --pred-dir outputs/pred/${m}_${h}_post --workers 24 > logs/e106b_feat_${m}_${h}_test.log 2>&1
done; done
echo done > logs/e106b_features.done
