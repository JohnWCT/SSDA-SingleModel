"""Internal data contracts for the multi-label SSDA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class DrugIndex:
    drug_ids: tuple[str, ...]
    drug_to_index: dict[str, int]
    index_to_drug: dict[int, str]

    @property
    def n_drugs(self) -> int:
        return len(self.drug_ids)


@dataclass(frozen=True)
class OmicsTable:
    x: NDArray[np.float32]
    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    domain: Literal["source", "target"]


@dataclass(frozen=True)
class ResponseMatrix:
    y: NDArray[np.float32]
    mask: NDArray[np.float32]
    sample_ids: tuple[str, ...]
    drug_index: DrugIndex
    domain: Literal["source", "target"]


@dataclass(frozen=True)
class TargetMasks:
    observed_mask: NDArray[np.float32]
    labeled_mask: NDArray[np.float32]
    unlabeled_mask: NDArray[np.float32]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SourceFold:
    fold_id: int
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


@dataclass(frozen=True)
class PreparedData:
    source_omics: OmicsTable
    target_omics: OmicsTable
    source_response: ResponseMatrix
    target_response: ResponseMatrix
    target_masks: TargetMasks
    drug_index: DrugIndex
    folds: tuple[SourceFold, ...]
    source_test_indices: tuple[int, ...]
    cancer_type_df: pd.DataFrame | None
    cancer_type_summary: pd.DataFrame
    resolved_source_cancer_type_path: str | None
    resolved_target_cancer_type_path: str | None
    alignment_report: pd.DataFrame
    nshot_summary: pd.DataFrame
