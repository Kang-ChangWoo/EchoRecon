#!/bin/bash
# E103: wait for the posterior prediction pass, then rank B's voxels by confidence vs support.
REPO=$(cd "$(dirname "$0")/.." && pwd); PY=/opt/conda/envs/shared_audio/bin/python; cd "$REPO"
until [ -f logs/post_pred.done ]; do sleep 30; done
for m in r2 r8; do
  $PY src/baseline_support.py --mode $m --pred-dir outputs/pred/${m}_post --tag _post --split test --workers 20 --out results/E103_confidence > logs/e103_${m}_test.log 2>&1
  $PY src/baseline_support.py --mode $m --pred-dir outputs/pred/${m}_post --tag _post --split val --workers 20 --out results/E103_confidence > logs/e103_${m}_val.log 2>&1
done
echo done > logs/e103.done
