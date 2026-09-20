#!/bin/bash
# (a) front-end time resolution: retrain the base model (imported from the sibling
# checkout, never copied) with a shorter STFT window, everything else as its
# released r2 recipe (40 ep, bs 24, lr 5e-4, seed 0, n_fft 512 zero-padded).
# hop 32 for every retrained window so hop <= win/2 holds at win 64.
#   bash src/train_window.sh <win> <hop> <gpu> [nviews] [mode]
WIN=$1; HOP=$2; GPU=$3; NV=${4:-2}; MODE=${5:-r2}
REPO=$(cd "$(dirname "$0")/.." && pwd)
BASE=${ECHO_DEPTH_ROOT:-$REPO/../hear360}
RUN=oaa_${MODE}_w${WIN}_h${HOP}
cd "$BASE"
CUDA_VISIBLE_DEVICES=$GPU DATA_MODULE=data_0422 REPLICA_ROOT=/root/storage/replica_0422 R0422_SPLIT=off3 \
STFT_WIN=$WIN STFT_HOP=$HOP REPLICA_SPEC_CACHE= \
/opt/conda/envs/shared_audio/bin/python train_oaa.py --run-name $RUN --nviews $NV --data-mode $MODE \
  --epochs 40 --batch-size 12 --accum 2 --lr 5e-4 --num-workers 8 --seed 0 --out-dir "$REPO/outputs/base_retrain" \
  > "$REPO/logs/$RUN.log" 2>&1
