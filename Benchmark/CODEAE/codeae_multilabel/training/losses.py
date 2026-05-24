"""Masked supervised losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def safe_masked_mean(raw_loss: Tensor, mask: Tensor) -> Tensor:
    denom = mask.sum().clamp_min(1.0)
    out: Tensor = (raw_loss * mask).sum() / denom
    return out


def masked_bce_with_logits(logits: Tensor, y: Tensor, mask: Tensor) -> Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return safe_masked_mean(raw, mask)


def masked_mae(pred: Tensor, y: Tensor, mask: Tensor) -> Tensor:
    raw = torch.abs(pred - y)
    return safe_masked_mean(raw, mask)
