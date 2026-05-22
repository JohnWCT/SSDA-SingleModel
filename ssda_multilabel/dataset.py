"""PyTorch datasets for multi-label SSDA."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class MultiLabelSampleDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        sample_ids: list[str],
        labeled_mask: torch.Tensor | None = None,
        unlabeled_mask: torch.Tensor | None = None,
    ) -> None:
        self.x = x
        self.y = y
        self.mask = mask
        self.sample_ids = sample_ids
        self.labeled_mask = labeled_mask
        self.unlabeled_mask = unlabeled_mask

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        item: dict[str, torch.Tensor | str | int] = {
            "x": self.x[idx],
            "y": self.y[idx],
            "mask": self.mask[idx],
            "sample_id": self.sample_ids[idx],
            "index": idx,
        }
        if self.labeled_mask is not None:
            item["labeled_mask"] = self.labeled_mask[idx]
        if self.unlabeled_mask is not None:
            item["unlabeled_mask"] = self.unlabeled_mask[idx]
        return item
