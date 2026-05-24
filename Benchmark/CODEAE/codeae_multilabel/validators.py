"""Input validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from codeae_multilabel.contracts import DrugIndex, ResponseMatrix, SourceFold


def validate_omics_table(df: pd.DataFrame, sample_id_col: str) -> None:
    if sample_id_col not in df.columns:
        raise ValueError(f"missing sample column: {sample_id_col}")
    sids = df[sample_id_col].astype(str).str.strip()
    if sids.duplicated().any():
        raise ValueError("duplicate sample IDs in omics table")
    feat = df.drop(columns=[sample_id_col])
    non_num = [c for c in feat.columns if not pd.api.types.is_numeric_dtype(feat[c])]
    if non_num:
        raise ValueError(f"non-numeric feature columns: {non_num[:5]}")


def validate_response_long_table(
    df: pd.DataFrame,
    sample_id_col: str,
    drug_col: str,
    response_col: str,
    task_type: str,
    domain: str,
) -> None:
    for col in (sample_id_col, drug_col, response_col):
        if col not in df.columns:
            raise ValueError(f"missing column {col!r} in {domain} response")
    drugs = df[drug_col].astype(str).str.strip()
    if (drugs == "").any():
        raise ValueError(f"empty drug_id in {domain} response")
    resp = pd.to_numeric(df[response_col], errors="coerce")
    if resp.isna().all():
        raise ValueError(f"response column {response_col!r} is not numeric")
    if domain == "target":
        valid = resp.dropna()
        if not set(valid.unique()).issubset({0.0, 1.0}):
            raise ValueError("target response must be binary 0/1")
    if task_type == "classification" and domain == "source":
        valid = resp.dropna()
        if len(valid) and not set(valid.unique()).issubset({0.0, 1.0}):
            raise ValueError("source classification response must be binary 0/1")


def validate_drug_index(drug_index: DrugIndex) -> None:
    if not drug_index.drug_ids:
        raise ValueError("drug index is empty")
    if len(drug_index.drug_ids) != len(drug_index.drug_to_index):
        raise ValueError("drug index has duplicate drug ids")


def validate_response_matrix(matrix: ResponseMatrix) -> None:
    if matrix.y.shape != matrix.mask.shape:
        raise ValueError("y and mask shape mismatch")
    if matrix.y.shape[0] != len(matrix.sample_ids):
        raise ValueError("sample count mismatch")
    if matrix.y.shape[1] != matrix.drug_index.n_drugs:
        raise ValueError("drug dimension mismatch")
    uniq = set(np.unique(matrix.mask))
    if not uniq.issubset({0.0, 1.0}):
        raise ValueError(f"mask must be 0/1, got {uniq}")


def validate_folds(folds: list[SourceFold]) -> None:
    for fold in folds:
        tr, va, te = set(fold.train_sample_ids), set(fold.val_sample_ids), set(fold.test_sample_ids)
        if tr & va or tr & te or va & te:
            raise ValueError(f"fold {fold.fold_id} has overlapping splits")
        if not tr or not va:
            raise ValueError(f"fold {fold.fold_id} has empty train or val")
