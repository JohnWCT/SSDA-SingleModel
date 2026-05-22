"""Source sample-level test split and K-fold CV."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold, train_test_split

from ssda_multilabel.schemas import SourceFold


def split_source_samples(
    n_samples: int,
    y_source: NDArray[np.float32],
    mask_source: NDArray[np.float32],
    test_size: float,
    n_splits: int,
    seed: int,
) -> tuple[tuple[int, ...], tuple[SourceFold, ...]]:
    indices = np.arange(n_samples)
    # Sample-level pseudo-label for stratification: majority class over observed drugs
    pseudo = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        obs = mask_source[i] > 0
        if obs.any():
            vals = y_source[i, obs]
            pseudo[i] = int(np.round(vals.mean()) >= 0.5)
    try:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=pseudo,
        )
    except ValueError:
        train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    test_idx = tuple(int(i) for i in sorted(test_idx))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[SourceFold] = []
    for fold_id, (tr, va) in enumerate(kf.split(train_val_idx)):
        train_indices = tuple(int(train_val_idx[i]) for i in tr)
        val_indices = tuple(int(train_val_idx[i]) for i in va)
        folds.append(
            SourceFold(
                fold_id=fold_id,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_idx,
            )
        )
    return test_idx, tuple(folds)
