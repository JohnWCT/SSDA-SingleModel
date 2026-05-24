"""Bridge to legacy CODE-AE auxiliary losses (optional)."""

from __future__ import annotations

from typing import Any

from torch import Tensor


class CodeAELossAdapter:
    def compute_pretrain_losses(
        self, source_batch: dict[str, Any], target_batch: dict[str, Any], model: Any
    ) -> dict[str, Tensor]:
        del source_batch, target_batch, model
        return {}

    def compute_finetune_losses(
        self, source_batch: dict[str, Any], target_batch: dict[str, Any], model: Any
    ) -> dict[str, Tensor]:
        del source_batch, target_batch, model
        return {}
