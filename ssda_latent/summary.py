"""Aggregate metrics across folds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_fold_csv(run_dir: Path, fold: int, name: str) -> pd.DataFrame:
    path = run_dir / f"fold_{fold}" / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def aggregate_fold_metrics(run_dir: Path, n_splits: int) -> pd.DataFrame:
    rows = []
    for fold in range(n_splits):
        for name, col in [
            ("source_val_metrics.csv", "source_val"),
            ("source_test_metrics.csv", "source_test"),
            ("target_prediction_metrics.csv", "target"),
        ]:
            df = _read_fold_csv(run_dir, fold, name)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            row["fold"] = fold
            row["metric_group"] = col
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_latent_metrics(run_dir: Path, n_splits: int) -> pd.DataFrame:
    rows = []
    for fold in range(n_splits):
        df = _read_fold_csv(run_dir, fold, "latent_distribution_metrics.csv")
        if not df.empty:
            rows.append(df.iloc[0].to_dict())
    return pd.DataFrame(rows)


def aggregate_kmeans_metrics(run_dir: Path, n_splits: int) -> pd.DataFrame | None:
    rows = []
    for fold in range(n_splits):
        path = run_dir / f"fold_{fold}" / "kmeans_cancer_type_metrics.csv"
        if path.exists():
            rows.append(pd.read_csv(path).iloc[0].to_dict())
    if not rows:
        return None
    return pd.DataFrame(rows)
