"""Target position-level n-shot labeled / unlabeled masks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssda_multilabel.schemas import TargetMasks


def build_target_nshot_masks(
    y_target: NDArray[np.float32],
    observed_mask: NDArray[np.float32],
    n_shot: int,
    seed: int,
) -> TargetMasks:
    n_samples, n_drugs = observed_mask.shape
    labeled = np.zeros_like(observed_mask, dtype=np.float32)
    warnings: list[str] = []
    rng = np.random.default_rng(seed)

    for d in range(n_drugs):
        obs = observed_mask[:, d] > 0
        if not obs.any():
            warnings.append(f"drug_index={d}: no target labels, skip n-shot")
            continue
        for cls in (0.0, 1.0):
            positions = np.where(obs & (y_target[:, d] == cls))[0]
            if len(positions) == 0:
                warnings.append(f"drug_index={d} class={int(cls)}: no positions")
                continue
            k = min(n_shot, len(positions))
            if k < n_shot:
                warnings.append(
                    f"drug_index={d} class={int(cls)}: only {len(positions)} < n_shot={n_shot}"
                )
            chosen = rng.choice(positions, size=k, replace=False)
            labeled[chosen, d] = 1.0

    unlabeled = observed_mask - labeled
    unlabeled = np.clip(unlabeled, 0.0, 1.0)
    overlap = (labeled > 0) & (unlabeled > 0)
    if overlap.any():
        raise RuntimeError("labeled and unlabeled masks overlap")
    if not np.allclose(labeled + unlabeled, observed_mask, atol=1e-6):
        raise RuntimeError("labeled + unlabeled != observed")

    return TargetMasks(
        observed_mask=observed_mask.astype(np.float32),
        labeled_mask=labeled,
        unlabeled_mask=unlabeled.astype(np.float32),
        warnings=tuple(warnings),
    )
