#!/bin/bash
# (a) chain: wait for a window retrain to finish, predict every test/val sequence with it,
# then run the cause decomposition on those predictions (test and val).
#   bash src/e100_after_train.sh <run e.g. oaa_r2_w128_h32> <gpu> <mode>
RUN=$1; GPU=$2; MODE=${3:-r2}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=/opt/conda/envs/shared_audio/bin/python
NAME=${RUN#oaa_}            # r2_w128_h32
until [ -f "$REPO/outputs/base_retrain/$RUN/train_done.json" ]; do sleep 60; done
cd "$REPO"
$PY src/predict_all.py --mode $MODE --ckpt outputs/base_retrain/$RUN/best.pth --name $NAME --gpu $GPU > logs/predict_$NAME.log 2>&1
$PY src/cause_decomp.py --mode $MODE --pred-dir outputs/pred/$NAME --split test --tag _${NAME#${MODE}_} --workers 24 > logs/e100_${NAME}_test.log 2>&1
$PY src/cause_decomp.py --mode $MODE --pred-dir outputs/pred/$NAME --split val --tag _${NAME#${MODE}_} --workers 24 > logs/e100_${NAME}_val.log 2>&1
echo done > logs/e100_${NAME}.done
