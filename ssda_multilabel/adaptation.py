"""Target unlabeled adaptation losses (multi-label binary entropy)."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from ssda_multilabel.losses import safe_masked_mean


def masked_entropy_loss(logits: Tensor, mask: Tensor) -> Tensor:
    """Per-drug binary entropy on sigmoid probabilities, averaged over masked positions."""
    p = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return safe_masked_mean(ent, mask)


def masked_adentropy_loss(logits: Tensor, mask: Tensor, eta: float = 0.1) -> Tensor:
    """Negative entropy weighted adaptation (maximize entropy on unlabeled positions)."""
    p = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return cast(Tensor, safe_masked_mean(ent, mask) * (-eta))
