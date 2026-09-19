# Related work, as it bears on this project

Notes taken from the papers themselves, with the numbers as published. Kept so
that positioning and evaluation protocol decisions are made against what exists.

## DAPS — Yun, Na, Kim, *Dense 2D-3D Indoor Prediction with Sound via Aligned Cross-Modal Distillation*, ICCV 2023 (Seoul National University)

Paper: openaccess.thecvf.com/content/ICCV2023/papers/Yun_Dense_2D-3D_Indoor_Prediction_with_Sound_via_Aligned_Cross-Modal_Distillation_ICCV_2023_paper.pdf
Code: github.com/hs-yn/DAPS

**Task and input.** Three dense predictions from audio alone at *one* receiver
position: equirectangular depth, equirectangular semantic segmentation, and 3D
scene reconstruction. Input is binaural audio, sweep-convolved, with the emitter
and the receiver at the same coordinate. One observation per sample; no
trajectory, no fusion across positions, and the receiver pose is not an input
because everything is expressed in the receiver's own frame.

**Method.** Vision-to-audio cross-modal distillation. A vision teacher (DPT /
U-Net for 2D, ConvONet for 3D) is trained on ground truth, and the audio student
is trained to match it through SAM (Spatial Alignment via Matching) blocks that
align multi-scale latent feature maps between modalities of different shape.

**Data.** DAPS, their own benchmark: 15.8K observations built from Matterport3D
with SoundSpaces. Each sample is binaural audio, an RGB panorama and a 3D voxel
volume. The 3D ground truth is the scene mesh truncated to a 2.5 x 2.5 x 2 m box
around the receiver, cleaned by clustering-based filtering and voxelised at
32^3. Samples with weak audio, more than 10 % missing 2D labels, or corrupted
voxels are excluded.

**3D numbers (DAPS-3D test split).**

| | IoU ↑ | Chamfer-L1 ↓ | NC ↑ | F1 ↑ |
|---|---|---|---|---|
| vision teacher (ConvONet) | 0.548 | 0.0137 | 0.882 | 0.560 |
| audio only, mono | 0.126 | 0.0698 | 0.625 | 0.189 |
| audio only, stereo | 0.136 | 0.0643 | 0.639 | 0.196 |
| best distilled (SAM full, U-Net / ViT) | 0.178 | 0.0555 / 0.0587 | 0.679 / 0.682 | 0.203 / 0.204 |

2D depth is reported as MAE, RMSE and delta accuracies; semantics as pixel
accuracy, mean accuracy, mIoU and a three-class layout IoU.

**What this means here.**

- The 3D task is a receiver-centred 2.5 x 2.5 x 2 m box at one position. Fusing
  a posed trajectory into a room-scale model is a different question, and the
  volume our evaluation covers has to be stated rather than assumed comparable.
- Audio-only 3D is far from the visual ceiling even inside that small box: the
  best audio result is IoU 0.178 against the vision teacher's 0.548. Any
  room-scale number we report should be read against that, not against vision.
- They report no oracle or upper bound. The ceiling measured in `README.md`
  (how much of the fused point set lies near the true surface) is not something
  this line of work has reported, which is where our diagnosis sits.
- Their protocol is the one to match for the paper's 3D table: mesh-derived
  voxel occupancy with IoU, Chamfer-L1, normal consistency and F1, per
  `DIRECTION.md`.
- The paper claims to be "the first to tackle dense indoor prediction of
  omnidirectional surroundings in both 2D and 3D with audio observations". That
  claim is taken; we do not make one.
- The paper states no limitations section; what is out of its scope (multiple
  positions, pose, room-scale volume) is inferred from the method, not quoted.
