"""PyTorch datasets for multilabel CODE-AE."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class MultiDrugSampleDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        x: NDArray[np.float32],
        y: NDArray[np.float32] | None = None,
        mask: NDArray[np.float32] | None = None,
        sample_ids: list[str] | None = None,
        domain: str | None = None,
    ) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)
        self.mask = None if mask is None else torch.as_tensor(mask, dtype=torch.float32)
        self.sample_ids = sample_ids or [str(i) for i in range(len(x))]
        self.domain = domain

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {"x": self.x[idx], "sample_id": self.sample_ids[idx]}
        if self.y is not None:
            item["y"] = self.y[idx]
        if self.mask is not None:
            item["mask"] = self.mask[idx]
        if self.domain is not None:
            item["domain"] = self.domain
        return item
