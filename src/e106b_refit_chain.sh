#!/bin/bash
# E106b refit path: train-scene predictions and features for the warm / head-only heads of one mode.
#   bash src/e106b_refit_chain.sh <mode> <gpu>
MODE=$1; GPU=$2
REPO=$(cd "$(dirname "$0")/.." && pwd); PY=/opt/conda/envs/shared_audio/bin/python; cd "$REPO"
for h in warm warm_headonly; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY src/posterior_predict.py --run posterior_${MODE}_${h} --mode $MODE --gpu $GPU --name ${MODE}_${h}_post --split train > logs/post_pred_${MODE}_${h}_train.log 2>&1 || { echo "predict ${MODE}_${h} failed" >> logs/e106b_refit_${MODE}.failed; exit 1; }
  $PY src/voxel_features.py --mode ${MODE}_${h} --split train --pred-dir outputs/pred/${MODE}_${h}_post --workers 16 > logs/e106b_feat_${MODE}_${h}_train.log 2>&1 || { echo "features ${MODE}_${h} failed" >> logs/e106b_refit_${MODE}.failed; exit 1; }
  # the refit learner also needs val features of the same head for the grid choice
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY src/posterior_predict.py --run posterior_${MODE}_${h} --mode $MODE --gpu $GPU --name ${MODE}_${h}_post > logs/post_pred_${MODE}_${h}_heldout.log 2>&1
  $PY src/voxel_features.py --mode ${MODE}_${h} --split val --pred-dir outputs/pred/${MODE}_${h}_post --workers 16 > logs/e106b_feat_${MODE}_${h}_val.log 2>&1
done
echo done > logs/e106b_refit_${MODE}.done
