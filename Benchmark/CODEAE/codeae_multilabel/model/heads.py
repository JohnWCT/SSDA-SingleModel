"""Multi-output drug prediction head."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class MultiOutputDrugHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        n_drugs: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_drugs))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        out: Tensor = self.net(z)
        return out
