#!/bin/bash
# issue A: after the warm-start posterior head finishes, fit its temperature on val (ray eval),
# then Stage D v3 under both references (test for numbers, val for the kept fraction).
#   bash src/issueA_chain.sh <mode> <gpu>
MODE=$1; GPU=$2; SUF=${3:-warm}; RUN=posterior_${MODE}_${SUF}
REPO=$(cd "$(dirname "$0")/.." && pwd); PY=/opt/conda/envs/shared_audio/bin/python; cd "$REPO"
until [ -f outputs/posterior/$RUN/train_done.json ]; do sleep 60; done
$PY src/eval_posterior_rays.py --run $RUN --split test --gpu $GPU --fit-temperature --batch-size 2 > logs/rays_${RUN}.log 2>&1
for ref in subset fixed; do
  for split in test val; do
    tag=_${SUF}_v3; [ "$ref" = fixed ] && tag=_${SUF}_v3fixed
    $PY src/stage_d.py --run $RUN --mode $MODE --gpu $GPU --split $split --ref $ref --tag $tag > logs/stage_d_${MODE}_${split}_${ref}_warm.log 2>&1 &
  done
done
wait
echo done > logs/issueA_${MODE}_${SUF}.done
