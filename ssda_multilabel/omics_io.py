"""Omics and response table loading with feature alignment."""

from __future__ import annotations

import pandas as pd

from ssda_multilabel.schemas import OmicsTable


def _resolve_col(df: pd.DataFrame, col: str) -> str:
    if col in df.columns:
        return col
    lookup = {str(c).lower(): c for c in df.columns}
    key = col.lower()
    if key in lookup:
        return lookup[key]
    raise ValueError(f"column {col!r} not found in {list(df.columns)}")


def read_omics_table(path: str, sample_id_col: str, domain: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    sid = _resolve_col(df, sample_id_col)
    df = df.copy()
    df[sid] = df[sid].astype(str).str.strip()
    if df[sid].duplicated().any():
        raise ValueError(f"duplicate sample IDs in omics table: {path}")
    feature_cols = [c for c in df.columns if c != sid]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[feature_cols].isna().any().any():
        raise ValueError(f"non-numeric omics features in {path}")
    df = df.set_index(sid)
    return df


def align_omics_features(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[OmicsTable, OmicsTable, pd.DataFrame]:
    common = sorted(set(source_df.columns) & set(target_df.columns))
    src_only = sorted(set(source_df.columns) - set(target_df.columns))
    tgt_only = sorted(set(target_df.columns) - set(source_df.columns))
    if not common:
        raise ValueError("no common omics features between source and target")
    src_ids = tuple(source_df.index.astype(str).tolist())
    tgt_ids = tuple(target_df.index.astype(str).tolist())
    src_x = source_df[common].to_numpy(dtype="float32")
    tgt_x = target_df[common].to_numpy(dtype="float32")
    report = pd.DataFrame(
        [
            {"metric": "n_common_features", "value": len(common)},
            {"metric": "n_source_only_features", "value": len(src_only)},
            {"metric": "n_target_only_features", "value": len(tgt_only)},
            {"metric": "n_source_samples", "value": len(src_ids)},
            {"metric": "n_target_samples", "value": len(tgt_ids)},
        ]
    )
    return (
        OmicsTable(x=src_x, sample_ids=src_ids, feature_names=tuple(common), domain="source"),
        OmicsTable(x=tgt_x, sample_ids=tgt_ids, feature_names=tuple(common), domain="target"),
        report,
    )


def read_response_long(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
