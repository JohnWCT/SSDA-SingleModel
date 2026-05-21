"""Encoder bottleneck latent extraction."""

from __future__ import annotations

from typing import cast

import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


def get_encoder_latent(encoder: nn.Module, x: Tensor) -> Tensor:
    out = encoder(x)
    if isinstance(out, tuple):
        return cast(Tensor, out[0])
    return cast(Tensor, out)


def encode_latent_dict(
    encoder: nn.Module,
    x_df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> dict[str, list[float]]:
    encoder.eval()
    sample_ids = x_df.index.astype(str).tolist()
    x_tensor = torch.FloatTensor(x_df.values)
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents: dict[str, list[float]] = {}
    offset = 0
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            z = get_encoder_latent(encoder, batch_x)
            z_cpu = z.cpu().numpy()
            for i in range(z_cpu.shape[0]):
                sid = sample_ids[offset + i]
                latents[sid] = z_cpu[i].tolist()
            offset += z_cpu.shape[0]
    return latents
