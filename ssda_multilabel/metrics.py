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
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

CLASSIFICATION_METRICS = (
    "auc",
    "aupr",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
)
REGRESSION_METRICS = ("mae", "rmse", "r2", "pearson", "spearman")


def _safe_auc(y_true: NDArray[np.int_], y_score: NDArray[np.float64]) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    scores = np.asarray(y_score, dtype=np.float64)
    if not np.isfinite(scores).all():
        return float("nan")
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def _safe_aupr(y_true: NDArray[np.int_], y_score: NDArray[np.float64]) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    scores = np.asarray(y_score, dtype=np.float64)
    if not np.isfinite(scores).all():
        return float("nan")
    try:
        return float(average_precision_score(y_true, scores))
    except ValueError:
        return float("nan")


def _classification_scores(g: pd.DataFrame) -> NDArray[np.float64]:
    """Ranking scores for AUC/AUPR: probability if present, else pred_score."""
    prob = g["probability"].astype(float)
    if prob.notna().any():
        return prob.fillna(g["pred_score"].astype(float)).to_numpy(dtype=np.float64)
    return g["pred_score"].astype(float).to_numpy(dtype=np.float64)


def compute_classification_metrics_per_drug(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for drug_id, g in df.groupby("drug_id"):
        y = np.asarray(g["ground_truth"].astype(int).values)
        y_score = _classification_scores(g)
        pred = np.asarray(g["pred_label"].astype(int).values)
        rows.append(
            {
                "drug_id": drug_id,
                "n": len(g),
                "auc": _safe_auc(y, y_score),
                "aupr": _safe_aupr(y, y_score),
                "accuracy": float(accuracy_score(y, pred)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            }
        )
    return pd.DataFrame(rows)


def _metrics_summary(
    per_drug: pd.DataFrame,
    metrics: tuple[str, ...],
    overall: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for m in metrics:
        vals = per_drug[m].dropna()
        row: dict[str, object] = {
            "metric": m,
            "macro": float(vals.mean()) if len(vals) else float("nan"),
        }
        w = np.asarray(per_drug["n"].values, dtype=np.float64)
        v = np.asarray(per_drug[m].values, dtype=np.float64)
        mask = ~np.isnan(v)
        if mask.any():
            row["weighted"] = float(np.average(v[mask], weights=w[mask]))
        else:
            row["weighted"] = float("nan")
        if overall is not None:
            row["overall"] = overall.get(m, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_classification_metrics_overall(df: pd.DataFrame) -> dict[str, float]:
    """Pooled metrics: all observed sample-drug pairs as one set."""
    if df.empty:
        return {m: float("nan") for m in CLASSIFICATION_METRICS}
    y = np.asarray(df["ground_truth"].astype(int).values)
    y_score = _classification_scores(df)
    pred = np.asarray(df["pred_label"].astype(int).values)
    return {
        "auc": _safe_auc(y, y_score),
        "aupr": _safe_aupr(y, y_score),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def compute_classification_metrics_summary(
    per_drug: pd.DataFrame,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    overall = compute_classification_metrics_overall(df) if df is not None else None
    return _metrics_summary(per_drug, CLASSIFICATION_METRICS, overall)


def compute_regression_metrics_per_drug(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for drug_id, g in df.groupby("drug_id"):
        y = np.asarray(g["ground_truth"].astype(float).values)
        p = np.asarray(g["pred_score"].astype(float).values)
        mae = float(mean_absolute_error(y, p))
        rmse = float(np.sqrt(mean_squared_error(y.tolist(), p.tolist())))
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
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "pearson": float(pr) if pr == pr else float("nan"),
                "spearman": float(sp) if sp == sp else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def compute_regression_metrics_overall(df: pd.DataFrame) -> dict[str, float]:
    """Pooled metrics: all observed sample-drug pairs as one set."""
    if df.empty:
        return {m: float("nan") for m in REGRESSION_METRICS}
    y = np.asarray(df["ground_truth"].astype(float).values)
    p = np.asarray(df["pred_score"].astype(float).values)
    mae = float(mean_absolute_error(y, p))
    rmse = float(np.sqrt(mean_squared_error(y.tolist(), p.tolist())))
    r2 = float(r2_score(y, p)) if len(y) > 1 else float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_std = float(np.std(y.astype(np.float64)))
        pr = pearsonr(y, p)[0] if len(y) > 1 and y_std > 0 else float("nan")
        sp = spearmanr(y, p).correlation if len(y) > 1 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": float(pr) if pr == pr else float("nan"),
        "spearman": float(sp) if sp == sp else float("nan"),
    }


def compute_regression_metrics_summary(
    per_drug: pd.DataFrame,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    overall = compute_regression_metrics_overall(df) if df is not None else None
    return _metrics_summary(per_drug, REGRESSION_METRICS, overall)


def compute_metrics_from_predictions(
    df: pd.DataFrame, task_type: str, domain: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regression run: source=MAE/RMSE/...; target=binary Label -> classification metrics."""
    sub = df[df["domain"] == domain] if "domain" in df.columns else df
    if task_type == "regression" and domain == "source":
        per = compute_regression_metrics_per_drug(sub)
        summ = compute_regression_metrics_summary(per, sub)
        return per, summ
    per = compute_classification_metrics_per_drug(sub)
    summ = compute_classification_metrics_summary(per, sub)
    return per, summ
