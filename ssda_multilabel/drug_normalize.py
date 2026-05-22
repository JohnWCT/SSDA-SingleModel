"""Drug name helpers; PRISM CSV should already include drug_name (see scripts/)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ssda_multilabel.sample_id import DRUG_NAME_COL


def load_broad_id_to_name(map_path: str | None) -> dict[str, str]:
    if map_path is None or not Path(map_path).is_file():
        return {}
    df = pd.read_csv(map_path)
    id_col = "broad_id" if "broad_id" in df.columns else df.columns[0]
    name_col = "name" if "name" in df.columns else "drug_name"
    if name_col not in df.columns:
        raise ValueError(f"drug map file missing name column: {list(df.columns)}")
    out: dict[str, str] = {}
    for bid, name in zip(df[id_col].astype(str), df[name_col].astype(str)):
        out[bid] = name.strip().lower()
    return out


def normalize_drug_name_column(response_df: pd.DataFrame) -> pd.DataFrame:
    """Require pre-built ``drug_name`` column (run scripts/patch_dapl_csv_columns.py for PRISM)."""
    df = response_df.copy()
    if DRUG_NAME_COL not in df.columns:
        raise ValueError(
            f"response table must include {DRUG_NAME_COL!r}; "
            "patch PRISM with scripts/patch_dapl_csv_columns.py"
        )
    df[DRUG_NAME_COL] = df[DRUG_NAME_COL].astype(str).str.strip().str.lower()
    return df


def ensure_drug_name_column(
    response_df: pd.DataFrame,
    drug_id_col: str,
    map_path: str | None,
    broad_id_col: str = "broad_id",
) -> pd.DataFrame:
    """Backward-compatible alias: map broad_id when drug_name missing."""
    df = response_df.copy()
    if drug_id_col in df.columns:
        df[drug_id_col] = df[drug_id_col].astype(str).str.strip().str.lower()
        return df
    mapping = load_broad_id_to_name(map_path)
    if broad_id_col not in df.columns:
        raise ValueError(
            f"response table needs {drug_id_col!r} or {broad_id_col!r} for drug name mapping"
        )
    if not mapping:
        raise ValueError("drug_name_map_path required to convert broad_id to drug_name")
    df[drug_id_col] = df[broad_id_col].astype(str).map(mapping)
    missing = df[drug_id_col].isna().sum()
    if missing:
        raise ValueError(f"{missing} rows could not map broad_id to drug_name")
    return df
