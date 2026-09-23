#!/bin/bash
# E111 chain for one (mode, variant): wait for training (liveness-checked), predict heldout + train
# scenes, voxel features (test / val / train), E106b ranker without and with the head's aggregates,
# rankings with range bins (test and val).
#   bash src/e111_chain.sh <mode> <variant> <gpu>
MODE=$1; VAR=$2; GPU=$3; RUN=reliability_${MODE}_${VAR}; NAME=${MODE}_rel_${VAR}
REPO=$(cd "$(dirname "$0")/.." && pwd); PY=/opt/conda/envs/shared_audio/bin/python; cd "$REPO"
while [ ! -f outputs/reliability/$RUN/train_done.json ]; do
  pgrep -f "train_reliability.py --mode $MODE --variant $VAR" > /dev/null || { echo "trainer gone" > logs/e111_${MODE}_${VAR}.failed; exit 1; }; sleep 120; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY src/reliability_predict.py --run $RUN --mode $MODE --gpu $GPU --split all --name $NAME > logs/relpred_${NAME}.log 2>&1 || { echo "predict failed" > logs/e111_${MODE}_${VAR}.failed; exit 1; }
for sp in test val train; do $PY src/voxel_features.py --mode $NAME --split $sp --pred-dir outputs/pred/$NAME --workers 16 > logs/e111_feat_${NAME}_$sp.log 2>&1 || { echo "features $sp failed" > logs/e111_${MODE}_${VAR}.failed; exit 1; }; done
CUDA_VISIBLE_DEVICES=$GPU $PY src/e106b_learn.py --mode $NAME --eval ${NAME}_test --feats-fixed noconf --tag _ranker > logs/e111_ranker_${NAME}.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU $PY src/e106b_learn.py --mode $NAME --eval ${NAME}_test --feats-fixed all --tag _ranker_head > logs/e111_rankerhead_${NAME}.log 2>&1
$PY src/baseline_support.py --mode $MODE --pred-dir outputs/pred/$NAME --tag _rel_${VAR} --split test --bins --workers 16 --out results/E111_reliability > logs/e111_rank_${NAME}_test.log 2>&1
$PY src/baseline_support.py --mode $MODE --pred-dir outputs/pred/$NAME --tag _rel_${VAR} --split val --bins --workers 16 --out results/E111_reliability > logs/e111_rank_${NAME}_val.log 2>&1
echo done > logs/e111_${MODE}_${VAR}.done
