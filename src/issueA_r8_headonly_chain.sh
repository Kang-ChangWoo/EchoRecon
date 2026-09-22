#!/bin/bash
# r8 head-only: wait for training with a liveness check, then temperature fit and Stage D x4,
# each Stage D on the GPU with the most free memory at launch.
REPO=$(cd "$(dirname "$0")/.." && pwd); PY=/opt/conda/envs/shared_audio/bin/python; cd "$REPO"
RUN=posterior_r8_warm_headonly
while [ ! -f outputs/posterior/$RUN/train_done.json ]; do
  if ! pgrep -f train_r8_headonly_retry.sh > /dev/null; then echo "trainer gone without train_done: $(cat logs/train_${RUN}.status 2>/dev/null)" > logs/issueA_r8_warm_headonly.failed; exit 1; fi
  sleep 120
done
freest() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1; }
G=$(freest); CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY src/eval_posterior_rays.py --run $RUN --split test --gpu 0 --fit-temperature --batch-size 2 > logs/rays_${RUN}.log 2>&1
for ref in subset fixed; do for split in test val; do
  tag=_warm_headonly_v3; [ "$ref" = fixed ] && tag=_warm_headonly_v3fixed
  G=$(freest); CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY src/stage_d.py --run $RUN --mode r8 --gpu 0 --split $split --ref $ref --tag $tag > logs/stage_d_r8_${split}_${ref}_warm_headonly.log 2>&1 &
  sleep 240
done; done
wait
echo done > logs/issueA_r8_warm_headonly.done
