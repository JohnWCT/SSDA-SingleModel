"""Human-readable preparation reports and cross-fold metric aggregation."""

from __future__ import annotations

import pandas as pd

from ssda_multilabel.schemas import DrugIndex, TargetMasks


def build_nshot_summary(target_masks: TargetMasks) -> pd.DataFrame:
    rows = [{"warning": w} for w in target_masks.warnings]
    if not rows:
        rows = [{"warning": "none"}]
    labeled_per_drug = target_masks.labeled_mask.sum(axis=0)
    for j, n in enumerate(labeled_per_drug):
        rows.append({"warning": f"drug_col_{j}_labeled_count", "value": str(float(n))})
    return pd.DataFrame(rows)


def build_missing_data_report(
    source_mask_sum: float,
    target_mask_sum: float,
    n_drugs: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "source_observed_positions", "value": source_mask_sum},
            {"metric": "target_observed_positions", "value": target_mask_sum},
            {"metric": "n_drugs", "value": n_drugs},
        ]
    )


def drug_list_report(drug_index: DrugIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {"drug_id": list(drug_index.drug_ids), "drug_index": range(drug_index.n_drugs)}
    )


def _metric_value_columns(df: pd.DataFrame) -> list[str]:
    skip = {"drug_id", "n", "fold"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def aggregate_per_drug_metrics(fold_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean / std of per-drug metrics across folds."""
    if not fold_frames:
        return pd.DataFrame()
    combined = pd.concat(fold_frames, ignore_index=True)
    if "drug_id" not in combined.columns:
        return pd.DataFrame()
    metric_cols = _metric_value_columns(combined)
    rows: list[dict[str, object]] = []
    for drug_id, grp in combined.groupby("drug_id"):
        row: dict[str, object] = {"drug_id": drug_id, "n_folds": int(grp["fold"].nunique()) if "fold" in grp else len(grp)}
        for col in metric_cols:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_scalar_metrics(fold_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean / std for single-row metric tables (e.g. kmeans, latent)."""
    if not fold_frames:
        return pd.DataFrame()
    combined = pd.concat(fold_frames, ignore_index=True)
    cols = _metric_value_columns(combined)
    rows: list[dict[str, object]] = []
    for col in cols:
        vals = combined[col].dropna()
        rows.append(
            {
                "metric": col,
                "mean": float(vals.mean()) if len(vals) else float("nan"),
                "std": float(vals.std(ddof=0)) if len(vals) > 1 else float("nan"),
                "n_folds": len(vals),
            }
        )
    return pd.DataFrame(rows)


def aggregate_summary_metrics(fold_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Mean / std of summary metrics (macro / weighted) across folds."""
    if not fold_frames:
        return pd.DataFrame()
    combined = pd.concat(fold_frames, ignore_index=True)
    if "metric" not in combined.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for metric_name, grp in combined.groupby("metric"):
        row: dict[str, object] = {"metric": metric_name, "n_folds": len(grp)}
        for col in ("macro", "weighted"):
            if col not in grp.columns:
                continue
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
