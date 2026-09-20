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
# the base trainer does not record the STFT window / hop it was launched with (its data module
# reads them from the environment); write them into the checkpoint args so the base checkpoint
# builder and predict_all use the right front-end
WIN=$(echo $RUN | sed -E 's/.*_w([0-9]+)_h([0-9]+)/\1/'); HOP=$(echo $RUN | sed -E 's/.*_w([0-9]+)_h([0-9]+)/\2/')
$PY - <<PYEOF
import torch, glob
for f in glob.glob("outputs/base_retrain/$RUN/*.pth"):
    ck = torch.load(f, map_location="cpu", weights_only=False)
    ck["args"].update(stft_win=$WIN, stft_hop=$HOP, stft_nfft=512, stft_window=2799)
    torch.save(ck, f); print("patched", f, ck["args"]["stft_win"], ck["args"]["stft_hop"])
PYEOF
rm -rf outputs/pred/$NAME results/E100_cause/${NAME}* logs/e100_${NAME}.done
$PY src/predict_all.py --mode $MODE --ckpt outputs/base_retrain/$RUN/best.pth --name $NAME --gpu $GPU > logs/predict_$NAME.log 2>&1
$PY src/cause_decomp.py --mode $MODE --pred-dir outputs/pred/$NAME --split test --tag _${NAME#${MODE}_} --workers 24 > logs/e100_${NAME}_test.log 2>&1
$PY src/cause_decomp.py --mode $MODE --pred-dir outputs/pred/$NAME --split val --tag _${NAME#${MODE}_} --workers 24 > logs/e100_${NAME}_val.log 2>&1
echo done > logs/e100_${NAME}.done
