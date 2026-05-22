"""Sample-level deterministic latent export."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from ssda_multilabel.model import MultiLabelSSDAModel


def encode_latent_dict(
    model: MultiLabelSSDAModel,
    x: NDArray[np.float32],
    sample_ids: list[str],
    batch_size: int,
    device: str,
) -> dict[str, list[float]]:
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    out: dict[str, list[float]] = {}
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch_x = torch.from_numpy(x[start : start + batch_size]).to(dev)
            z = model.encode(batch_x, deterministic=True).cpu().numpy()
            for i, sid in enumerate(sample_ids[start : start + batch_size]):
                out[sid] = z[i].astype(float).tolist()
    return out
