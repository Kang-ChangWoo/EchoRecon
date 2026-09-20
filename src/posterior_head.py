"""A ray-wise categorical depth posterior on top of the base model's backbone.

The base model (imported from the sibling checkout, never copied here) ends in a
decoder whose last layer is a 1-channel convolution producing depth / max_depth
in [0, 1]. This wraps that model and replaces only the last layer with a
K-channel one, so the backbone, the attention stack and the decoder are exactly
the base model's and can be initialised from its checkpoint. The point head is
kept alongside, so the same network can report both and the comparison in
Stage D is within one backbone.

Bins are linear in metric depth over [d_min, d_max]; an active echo's time of
flight is linear in range.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_ROOT = Path(os.environ.get("ECHO_DEPTH_ROOT",
                                Path(__file__).resolve().parents[2] / "hear360")).resolve()


def load_base(args: dict):
    """Build the base model from a checkpoint's saved args, without copying its code."""
    os.environ.setdefault("DATA_MODULE", "data_0422")
    if str(BASE_ROOT) not in sys.path:
        sys.path.insert(0, str(BASE_ROOT))
    cwd = os.getcwd()
    os.chdir(BASE_ROOT)
    try:
        from core.ckpt import build
        from core.data import get_data_module
        DM = get_data_module()
        model, dmode, nch, kind, poses = build(args, DM)
    finally:
        os.chdir(cwd)
    return model, poses, DM


class PosteriorDepth(nn.Module):
    """Base model with a K-bin categorical head in place of the 1-channel one.

    forward(spec, view_poses) -> logits (B, K, H, W)
    point_depth(logits)       -> expected depth in metres
    argmax_depth(logits)      -> bin-centre depth in metres
    """

    def __init__(self, base: nn.Module, K: int = 128, d_min: float = 0.1, d_max: float = 10.0,
                 init_from_point_head: bool = True):
        super().__init__()
        self.base = base
        self.K = K
        self.d_min = float(d_min); self.d_max = float(d_max)
        old = base.head
        new = nn.Conv2d(old.in_channels, K, old.kernel_size, old.stride, old.padding)
        if init_from_point_head:
            # every bin starts from the point head's filter, so the initial
            # distribution is flat but already driven by the same features
            with torch.no_grad():
                new.weight.copy_(old.weight.repeat(K, 1, 1, 1) * 0.1)
                new.bias.zero_()
        self.base.head = new
        edges = torch.linspace(d_min, d_max, K + 1)
        self.register_buffer("edges", edges)
        self.register_buffer("centres", 0.5 * (edges[:-1] + edges[1:]))
        self._patch_decode()

    def _patch_decode(self):
        """The base decoder ends in sigmoid(head(x)); keep everything but that squash."""
        base = self.base
        orig = base._decode

        def decode(x_tok, fine_tok, poses):
            B = x_tok.size(0); dev = x_tok.device
            x = x_tok.transpose(1, 2).reshape(B, base.C, base.lh, base.lw)
            fine_erp = base._fine_lift(fine_tok, poses, dev)
            x = base.up_stages[0](x) + base.fine_to_dec(fine_erp)
            for st in base.up_stages[1:]:
                x = st(x)
            return base.head(x)                      # logits, no sigmoid
        base._decode = decode
        self._orig_decode = orig

    def forward(self, spec, view_poses=None):
        return self.base(spec, view_poses=view_poses)          # (B, K, H, W) logits

    def probs(self, logits):
        return F.softmax(logits, dim=1)

    def expected_depth(self, logits):
        p = self.probs(logits)
        return (p * self.centres.view(1, -1, 1, 1)).sum(1)

    def argmax_depth(self, logits):
        return self.centres[logits.argmax(1)]

    def bin_of(self, depth_m: torch.Tensor) -> torch.Tensor:
        step = (self.d_max - self.d_min) / self.K
        return ((depth_m - self.d_min) / step).floor().clamp(0, self.K - 1).long()

    def soft_target(self, depth_m: torch.Tensor, width_bins: float = 1.0) -> torch.Tensor:
        """Triangular/Gaussian soft label around the true depth, (B, K, H, W).

        width_bins = 0 gives a hard one-hot. A soft label keeps the head from
        being punished for a neighbouring bin, which a hard label over 128 bins
        does severely.
        """
        k = self.bin_of(depth_m).unsqueeze(1).float()
        ks = torch.arange(self.K, device=depth_m.device).view(1, -1, 1, 1).float()
        if width_bins <= 0:
            t = (ks == k).float()
        else:
            t = torch.exp(-0.5 * ((ks - k) / width_bins) ** 2)
        return t / t.sum(1, keepdim=True).clamp_min(1e-12)


def kl_loss(logits, target, mask):
    """KL(target || predicted) averaged over valid rays."""
    logp = F.log_softmax(logits, dim=1)
    kl = (target * (target.clamp_min(1e-12).log() - logp)).sum(1)
    m = mask.squeeze(1) if mask.dim() == 4 else mask
    return (kl * m).sum() / m.sum().clamp_min(1.0)
