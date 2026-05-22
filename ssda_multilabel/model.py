"""SSDA encoder + multi-output prediction head."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from model import DAE, MLP


class MultiOutputHead(nn.Module):
    def __init__(self, input_dim: int, n_drugs: int, hidden_dims: tuple[int, ...] = (32,)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3)])
            prev = h
        layers.append(nn.Linear(prev, n_drugs))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class MultiLabelSSDAModel(nn.Module):
    """Encoder (DAE/MLP) + multi-drug head; latent export uses deterministic encode path."""

    def __init__(
        self,
        encoder: nn.Module,
        head: MultiOutputHead,
        encoder_type: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.encoder_type = encoder_type

    def encode(self, x: Tensor, deterministic: bool = True) -> Tensor:
        if self.encoder_type == "dae":
            # Avoid stochastic denoising in DAE.forward
            return self.encoder.ae.encode(x)  # type: ignore[union-attr]
        return self.encoder(x)  # type: ignore[operator]

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x, deterministic=True)
        return self.head(z)


def build_model(
    input_dim: int,
    n_drugs: int,
    encoder: str,
    encoder_h_dims: tuple[int, ...],
    latent_dim: int = 128,
    dropout: float = 0.3,
) -> MultiLabelSSDAModel:
    h = list(encoder_h_dims)
    if encoder == "dae":
        enc = DAE(
            input_dim=latent_dim,
            AE_input_dim=input_dim,
            AE_h_dims=h,
            drop=dropout,
        )
        latent = latent_dim
    elif encoder == "mlp":
        enc = MLP(input_dim=input_dim, latent_dim=latent_dim, h_dims=h, drop_out=dropout)
        latent = latent_dim
    else:
        raise ValueError(f"unknown encoder: {encoder}")
    head = MultiOutputHead(latent, n_drugs, hidden_dims=(32,))
    return MultiLabelSSDAModel(enc, head, encoder)
