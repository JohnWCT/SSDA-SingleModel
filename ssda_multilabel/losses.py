"""Masked supervised losses for multi-label SSDA."""

from __future__ import annotations

import torch.nn.functional as torch_functional
from torch import Tensor


def safe_masked_mean(raw_loss: Tensor, mask: Tensor) -> Tensor:
    denom = mask.sum().clamp_min(1.0)
    out: Tensor = (raw_loss * mask).sum() / denom
    return out


def masked_bce_with_logits(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    raw = torch_functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return safe_masked_mean(raw, mask)


def masked_mse(pred: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    raw = torch_functional.mse_loss(pred, targets, reduction="none")
    return safe_masked_mean(raw, mask)


def masked_mae(pred: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    raw = torch_functional.l1_loss(pred, targets, reduction="none")
    return safe_masked_mean(raw, mask)


def masked_huber(pred: Tensor, targets: Tensor, mask: Tensor, delta: float = 1.0) -> Tensor:
    raw = torch_functional.huber_loss(pred, targets, reduction="none", delta=delta)
    return safe_masked_mean(raw, mask)


def masked_regression_loss(
    pred: Tensor,
    targets: Tensor,
    mask: Tensor,
    reg_loss: str,
) -> Tensor:
    if reg_loss == "mse":
        return masked_mse(pred, targets, mask)
    if reg_loss == "mae":
        return masked_mae(pred, targets, mask)
    if reg_loss == "huber":
        return masked_huber(pred, targets, mask)
    raise ValueError(f"unknown reg_loss: {reg_loss}")
