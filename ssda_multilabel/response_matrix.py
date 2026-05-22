"""Long response table -> wide matrix + mask."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ssda_multilabel.sample_id import sample_match_key
from ssda_multilabel.schemas import DrugIndex, ResponseMatrix


def _resolve_col(df: pd.DataFrame, col: str) -> str:
    if col in df.columns:
        return col
    lookup = {str(c).lower(): c for c in df.columns}
    if col.lower() in lookup:
        return lookup[col.lower()]
    raise ValueError(f"column {col!r} not found")


def long_to_response_matrix(
    response_df: pd.DataFrame,
    sample_ids: list[str],
    drug_index: DrugIndex,
    sample_id_col: str,
    drug_id_col: str,
    response_col: str,
    domain: str,
    duplicate_strategy: str = "error",
    *,
    omics_sample_id_col: str | None = None,
    response_sample_id_col: str | None = None,
) -> ResponseMatrix:
    sid_col = _resolve_col(response_df, sample_id_col)
    did_col = _resolve_col(response_df, drug_id_col)
    resp_col = _resolve_col(response_df, response_col)
    df = response_df[[sid_col, did_col, resp_col]].copy()
    df[sid_col] = df[sid_col].astype(str).str.strip()
    df[did_col] = df[did_col].astype(str)
    df[resp_col] = pd.to_numeric(df[resp_col], errors="coerce")

    n_samples = len(sample_ids)
    n_drugs = drug_index.n_drugs
    y = np.zeros((n_samples, n_drugs), dtype=np.float32)
    mask = np.zeros((n_samples, n_drugs), dtype=np.float32)
    omics_hint = omics_sample_id_col or sample_id_col
    resp_hint = response_sample_id_col or sample_id_col
    sample_to_row = {
        sample_match_key(sid, column_hint=omics_hint): i for i, sid in enumerate(sample_ids)
    }

    dup = df.duplicated(subset=[sid_col, did_col], keep=False)
    if dup.any():
        n_dup = int(dup.sum())
        if duplicate_strategy == "error":
            raise ValueError(
                f"duplicate sample-drug rows in {domain} response ({n_dup} rows); "
                "use --duplicate_response_strategy mean|first to collapse"
            )
        if duplicate_strategy == "first":
            df = df.drop_duplicates(subset=[sid_col, did_col], keep="first")
        elif duplicate_strategy == "mean":
            df = (
                df.groupby([sid_col, did_col], as_index=False)[resp_col]
                .mean()
                .rename(columns={resp_col: resp_col})
            )
        else:
            raise ValueError(f"unknown duplicate_strategy: {duplicate_strategy!r}")

    for _, row in df.iterrows():
        sid_key = sample_match_key(row[sid_col], column_hint=resp_hint)
        did = str(row[did_col]).strip().lower()
        if sid_key not in sample_to_row or did not in drug_index.drug_to_index:
            continue
        if pd.isna(row[resp_col]):
            continue
        i = sample_to_row[sid_key]
        j = drug_index.drug_to_index[did]
        y[i, j] = float(row[resp_col])
        mask[i, j] = 1.0

    return ResponseMatrix(
        y=y,
        mask=mask,
        sample_ids=tuple(sample_ids),
        drug_index=drug_index,
        domain=domain,  # type: ignore[arg-type]
    )
