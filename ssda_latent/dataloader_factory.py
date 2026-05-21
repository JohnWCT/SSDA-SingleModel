"""Build fold DataLoaders via utils.create_dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import utils
from ssda_latent.config import ExperimentConfig
from ssda_latent.data_loading import ExpressionTables
from ssda_latent.split import (
    SplitManifest,
    get_source_ids_for_fold,
    get_target_ids_by_role,
)


@dataclass(frozen=True)
class FoldDataLoaders:
    source: dict[str, DataLoader[Any]]
    target_labeled: dict[str, DataLoader[Any]]
    target_unlabeled: dict[str, DataLoader[Any]]


def _subset_xy(
    x: pd.DataFrame, y: pd.DataFrame, ids: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [str(i) for i in ids]
    x_sub = x.loc[ids]
    y_sub = y.loc[ids]
    return x_sub.T, y_sub


def _source_train_loader(
    x_source: pd.DataFrame,
    y_source: pd.DataFrame,
    train_ids: list[str],
    batch_size: int,
) -> DataLoader[Any]:
    x_tr, y_tr = _subset_xy(x_source, y_source, train_ids)
    class_sample_count = np.array(
        [
            Counter(y_tr["response"])[0] / len(y_tr["response"]),
            Counter(y_tr["response"])[1] / len(y_tr["response"]),
        ]
    )
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in y_tr["response"].values])
    samples_weight = torch.from_numpy(samples_weight).reshape(-1)
    sampler = WeightedRandomSampler(
        samples_weight.type(torch.DoubleTensor),
        len(samples_weight),
        replacement=True,
    )
    return cast(
        DataLoader[Any],
        utils.create_dataset(x=x_tr, y=y_tr, batch_size=batch_size, shuffle=False, sampler=sampler),
    )


def build_fold_dataloaders(
    tables: ExpressionTables,
    manifest: SplitManifest,
    fold_index: int,
    config: ExperimentConfig,
) -> FoldDataLoaders:
    bs = config.batch_size
    train_ids = get_source_ids_for_fold(manifest, fold_index, "source_fold_train")
    val_ids = get_source_ids_for_fold(manifest, fold_index, "source_fold_val")

    source_train = _source_train_loader(tables.x_source, tables.y_source, train_ids, bs)
    x_va, y_va = _subset_xy(tables.x_source, tables.y_source, val_ids)
    source_val = cast(
        DataLoader[Any],
        utils.create_dataset(x=x_va, y=y_va, batch_size=bs, shuffle=False),
    )

    x_lt, y_lt = _subset_xy(
        tables.x_target,
        tables.y_target,
        get_target_ids_by_role(manifest, "target_labeled_train"),
    )
    x_lv, y_lv = _subset_xy(
        tables.x_target,
        tables.y_target,
        get_target_ids_by_role(manifest, "target_labeled_val"),
    )
    x_ut, y_ut = _subset_xy(
        tables.x_target,
        tables.y_target,
        get_target_ids_by_role(manifest, "target_unlabeled_train"),
    )
    x_uv, y_uv = _subset_xy(
        tables.x_target,
        tables.y_target,
        get_target_ids_by_role(manifest, "target_unlabeled_val"),
    )

    return FoldDataLoaders(
        source={"train": source_train, "val": source_val},
        target_labeled={
            "train": cast(
                DataLoader[Any],
                utils.create_dataset(x=x_lt, y=y_lt, batch_size=bs, shuffle=True),
            ),
            "val": cast(
                DataLoader[Any],
                utils.create_dataset(x=x_lv, y=y_lv, batch_size=bs, shuffle=False),
            ),
        },
        target_unlabeled={
            "train": cast(
                DataLoader[Any],
                utils.create_dataset(x=x_ut, y=y_ut, batch_size=bs, shuffle=True),
            ),
            "val": cast(
                DataLoader[Any],
                utils.create_dataset(x=x_uv, y=y_uv, batch_size=bs, shuffle=False),
            ),
        },
    )
