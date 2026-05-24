"""Multilabel CODE-AE model wrapper."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from codeae_multilabel.model.heads import MultiOutputDrugHead
from codeae_multilabel.model.legacy_adapter import encode_with_codeae


class MultiLabelCodeAEModel(nn.Module):
    def __init__(self, codeae_core: nn.Module, prediction_head: MultiOutputDrugHead) -> None:
        super().__init__()
        self.codeae_core = codeae_core
        self.prediction_head = prediction_head

    def encode(self, x: Tensor, deterministic: bool = False) -> Tensor:
        return encode_with_codeae(self.codeae_core, x, deterministic=deterministic)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x, deterministic=False)
        out: Tensor = self.prediction_head(z)
        return out
