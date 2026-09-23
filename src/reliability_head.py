"""Per-ray reliability head on the released base model (E111). The decoder features feed the
original depth head and a new head that predicts P(|depth error| < tau) for the model's own depth,
with the predicted depth and the pixel elevation given explicitly."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityDepth(nn.Module):
    def __init__(self, base: nn.Module, hidden: int = 64):
        super().__init__()
        self.base = base
        C = base.head.in_channels
        self.rel = nn.Sequential(nn.Conv2d(C + 3, hidden, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.rel[-1].bias)
        self._patch()

    def _patch(self):
        base = self.base; outer = self

        def decode(x_tok, fine_tok, poses):
            B = x_tok.size(0); dev = x_tok.device
            x = x_tok.transpose(1, 2).reshape(B, base.C, base.lh, base.lw)
            fine_erp = base._fine_lift(fine_tok, poses, dev)
            x = base.up_stages[0](x) + base.fine_to_dec(fine_erp)
            for st in base.up_stages[1:]:
                x = st(x)
            depth = torch.sigmoid(base.head(x))                                   # (B,1,H,W) depth / max_depth
            H, W = depth.shape[-2:]
            v = (torch.arange(H, device=dev, dtype=depth.dtype) + 0.5) / H; phi = (0.5 - v) * math.pi
            s = torch.sin(phi).view(1, 1, H, 1).expand(B, 1, H, W); c = torch.cos(phi).view(1, 1, H, 1).expand(B, 1, H, W)
            rel_logit = outer.rel(torch.cat([x, depth.detach(), s, c], 1))
            outer._last_rel_logit = rel_logit
            return depth
        base._decode = decode

    def forward(self, spec, view_poses=None):
        """returns (depth in [0,1], reliability logit), both (B,1,H,W)"""
        d = self.base(spec, view_poses=view_poses)
        return d, self._last_rel_logit


def bce_masked(logit, target, mask, pos_weight):
    l = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight, reduction="none")
    return (l * mask).sum() / mask.sum().clamp_min(1.0)
