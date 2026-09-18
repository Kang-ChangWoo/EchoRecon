"""Forward-sequence loader: poses, ground-truth ERP depth, binaural echoes per step."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SEQ_ROOT = Path(os.environ.get("FORWARD_SEQ_ROOT", "/root/storage/replica_0422_forward_seq"))
RELS = ("000", "090", "180", "270")
TEST_SCENES = ("apartment_2", "frl_apartment_5", "office_4")
VAL_SCENES = ("apartment_1", "frl_apartment_4", "office_3")


def scenes():
    return sorted(d.name for d in SEQ_ROOT.iterdir() if d.is_dir())


def sequences(scene: str):
    return sorted(d.name for d in (SEQ_ROOT / scene).iterdir() if (d / "poses.json").exists())


class Sequence:
    def __init__(self, scene: str, seq: str):
        self.scene, self.seq = scene, seq
        self.dir = SEQ_ROOT / scene / seq
        self.poses = json.loads((self.dir / "poses.json").read_text())
        self.steps = [i for i in range(len(self.poses))
                      if (self.dir / "erp_depth_radial" / f"step_{i:03d}.npy").exists()]

    def __len__(self):
        return len(self.steps)

    def pose(self, i: int) -> dict:
        return self.poses[i]

    def gt_depth(self, i: int) -> np.ndarray:
        return np.load(self.dir / "erp_depth_radial" / f"step_{i:03d}.npy").astype(np.float32)

    def wav8(self, i: int, window: int) -> np.ndarray:
        """(8, window) float32: [000 L,R | 090 L,R | 180 L,R | 270 L,R], OAA's training order."""
        import soundfile as sf
        chans = []
        for rel in RELS:
            w, sr = sf.read(self.dir / "audio_wav" / f"step_{i:03d}_rel{rel}.wav", dtype="float32")
            assert sr == 48000 and w.ndim == 2 and w.shape[1] == 2, (sr, w.shape)
            w = w[:window].T
            if w.shape[1] < window:
                w = np.pad(w, ((0, 0), (0, window - w.shape[1])))
            chans.append(w)
        return np.concatenate(chans, 0)

    def rgb(self, i: int):
        from PIL import Image
        return np.asarray(Image.open(self.dir / "erp_rgb" / f"step_{i:03d}.png").convert("RGB"))
