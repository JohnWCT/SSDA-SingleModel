"""Per-drug and summary metrics for classification and regression."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def _safe_auc(y_true: NDArray[np.int_], y_score: NDArray[np.float64]) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def compute_classification_metrics_per_drug(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for drug_id, g in df.groupby("drug_id"):
        y = np.asarray(g["ground_truth"].astype(int).values)
        prob = np.asarray(g["probability"].astype(float).values)
        pred = np.asarray(g["pred_label"].astype(int).values)
        rows.append(
            {
                "drug_id": drug_id,
                "n": len(g),
                "auc": _safe_auc(y, prob),
                "aupr": float(average_precision_score(y, prob))
                if len(np.unique(y)) >= 2
                else float("nan"),
                "accuracy": float(accuracy_score(y, pred)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            }
        )
    return pd.DataFrame(rows)


def compute_classification_metrics_summary(per_drug: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auc", "aupr", "accuracy", "f1", "balanced_accuracy"]
    rows: list[dict[str, object]] = []
    for m in metrics:
        vals = per_drug[m].dropna()
        rows.append({"metric": m, "macro": float(vals.mean()) if len(vals) else float("nan")})
        w = np.asarray(per_drug["n"].values, dtype=np.float64)
        v = np.asarray(per_drug[m].values, dtype=np.float64)
        mask = ~np.isnan(v)
        if mask.any():
            rows[-1]["weighted"] = float(np.average(v[mask], weights=w[mask]))
        else:
            rows[-1]["weighted"] = float("nan")
    return pd.DataFrame(rows)


def compute_regression_metrics_per_drug(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for drug_id, g in df.groupby("drug_id"):
        y = np.asarray(g["ground_truth"].astype(float).values)
        p = np.asarray(g["pred_score"].astype(float).values)
        rmse = float(np.sqrt(mean_squared_error(y.tolist(), p.tolist())))
        mae = float(mean_absolute_error(y, p))
        r2 = float(r2_score(y, p)) if len(y) > 1 else float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_std = float(np.std(y.astype(np.float64)))
            pr = pearsonr(y, p)[0] if len(y) > 1 and y_std > 0 else float("nan")
            sp = spearmanr(y, p).correlation if len(y) > 1 else float("nan")
        rows.append(
            {
                "drug_id": drug_id,
                "n": len(g),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "pearson": float(pr) if pr == pr else float("nan"),
                "spearman": float(sp) if sp == sp else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def compute_regression_metrics_summary(per_drug: pd.DataFrame) -> pd.DataFrame:
    metrics = ["rmse", "mae", "r2", "pearson", "spearman"]
    rows = []
    for m in metrics:
        vals = per_drug[m].dropna()
        rows.append({"metric": m, "macro": float(vals.mean()) if len(vals) else float("nan")})
    return pd.DataFrame(rows)


def compute_metrics_from_predictions(
    df: pd.DataFrame, task_type: str, domain: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[df["domain"] == domain] if "domain" in df.columns else df
    if domain == "target" or task_type == "classification" or domain == "target":
        per = compute_classification_metrics_per_drug(sub)
        summ = compute_classification_metrics_summary(per)
        return per, summ
    if task_type == "regression" and domain == "source":
        per = compute_regression_metrics_per_drug(sub)
        summ = compute_regression_metrics_summary(per)
        return per, summ
    per = compute_classification_metrics_per_drug(sub)
    summ = compute_classification_metrics_summary(per)
    return per, summ
