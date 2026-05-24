"""DataLoader builders."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from codeae_multilabel.contracts import OmicsTable, ResponseMatrix
from codeae_multilabel.data.dataset import MultiDrugSampleDataset


def _omics_array(omics: OmicsTable) -> NDArray[np.float32]:
    out: NDArray[np.float32] = omics.x.loc[list(omics.sample_ids)].values.astype(np.float32)
    return out


def build_source_train_loader(
    omics: OmicsTable,
    response: ResponseMatrix,
    sample_ids: list[str],
    batch_size: int,
    seed: int,
) -> DataLoader[Any]:
    sid_to_idx = {s: i for i, s in enumerate(omics.sample_ids)}
    indices = [sid_to_idx[s] for s in sample_ids]
    x = _omics_array(omics)[indices]
    y = response.y[indices]
    mask = response.mask[indices]
    ds = MultiDrugSampleDataset(x, y, mask, list(sample_ids), "source")
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen, drop_last=False)


def build_source_eval_loader(
    omics: OmicsTable,
    response: ResponseMatrix,
    sample_ids: list[str],
    batch_size: int,
) -> DataLoader[Any]:
    sid_to_idx = {s: i for i, s in enumerate(omics.sample_ids)}
    indices = [sid_to_idx[s] for s in sample_ids]
    x = _omics_array(omics)[indices]
    y = response.y[indices]
    mask = response.mask[indices]
    ds = MultiDrugSampleDataset(x, y, mask, list(sample_ids), "source")
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def build_target_unlabeled_loader(
    omics: OmicsTable,
    batch_size: int,
    seed: int,
) -> DataLoader[Any]:
    x = _omics_array(omics)
    ds = MultiDrugSampleDataset(x, sample_ids=list(omics.sample_ids), domain="target")
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)


def build_target_eval_loader(
    omics: OmicsTable,
    response: ResponseMatrix,
    batch_size: int,
) -> DataLoader[Any]:
    x = _omics_array(omics)
    ds = MultiDrugSampleDataset(
        x, response.y, response.mask, list(omics.sample_ids), "target"
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
