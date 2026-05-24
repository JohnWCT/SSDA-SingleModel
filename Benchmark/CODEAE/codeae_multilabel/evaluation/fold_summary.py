"""Cross-fold metric aggregation aligned with multilabel SSDA outputs."""

from __future__ import annotations

import pandas as pd


def filter_source_test(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty or "split" not in pred.columns:
        return pred
    test = pred[pred["split"] == "source_test"]
    return test if len(test) else pred


def filter_target_eval(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pred
    if "split" in pred.columns:
        ev = pred[pred["split"] == "target_eval"]
        return ev if len(ev) else pred
    return pred


def normalize_per_drug_columns(per_drug: pd.DataFrame) -> pd.DataFrame:
    """Align column names with SSDA multilabel (auc, n)."""
    if per_drug.empty:
        return per_drug
    out = per_drug.copy()
    if "n_observed" in out.columns and "n" not in out.columns:
        out = out.rename(columns={"n_observed": "n"})
    if "auroc" in out.columns and "auc" not in out.columns:
        out = out.rename(columns={"auroc": "auc"})
    return out


def summary_to_ssda_wide(summary: pd.DataFrame) -> pd.DataFrame:
    """Convert long CODE-AE summary (aggregation rows) to SSDA wide format."""
    if summary.empty:
        return pd.DataFrame(columns=["metric", "macro", "weighted", "overall"])
    if "macro" in summary.columns and "metric" in summary.columns:
        return summary.copy()
    if "metric_name" not in summary.columns:
        return pd.DataFrame(columns=["metric", "macro", "weighted", "overall"])
    rows: list[dict[str, object]] = []
    for metric in summary["metric"].dropna().unique():
        sub = summary[summary["metric"] == metric]
        row: dict[str, object] = {"metric": _public_metric_name(str(metric))}
        for agg in ("macro", "weighted", "overall", "micro"):
            sel = sub[sub["aggregation"] == agg]
            if len(sel):
                row[agg] = float(sel["metric_value"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _public_metric_name(name: str) -> str:
    return "auc" if name == "auroc" else name


def _metric_value_columns(df: pd.DataFrame) -> list[str]:
    skip = {"drug_id", "n", "n_observed", "fold", "n_positive", "n_negative"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def aggregate_per_drug_metrics(fold_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not fold_frames:
        return pd.DataFrame()
    combined = pd.concat([normalize_per_drug_columns(f) for f in fold_frames], ignore_index=True)
    if "drug_id" not in combined.columns:
        return pd.DataFrame()
    metric_cols = _metric_value_columns(combined)
    rows: list[dict[str, object]] = []
    for drug_id, grp in combined.groupby("drug_id"):
        row: dict[str, object] = {
            "drug_id": drug_id,
            "n_folds": int(grp["fold"].nunique()) if "fold" in grp.columns else len(grp),
        }
        for col in metric_cols:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_scalar_metrics(fold_frames: list[pd.DataFrame]) -> pd.DataFrame:
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
    if not fold_frames:
        return pd.DataFrame()
    combined = pd.concat(fold_frames, ignore_index=True)
    if "metric" not in combined.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for metric_name, grp in combined.groupby("metric"):
        row: dict[str, object] = {"metric": metric_name, "n_folds": len(grp)}
        for col in ("macro", "weighted", "overall"):
            if col not in grp.columns:
                continue
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_combined_eval_summary(
    src_fold_frames: list[pd.DataFrame],
    tgt_fold_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if src_fold_frames:
        frames.append(aggregate_summary_metrics(src_fold_frames).assign(domain="source_test"))
    if tgt_fold_frames:
        frames.append(aggregate_summary_metrics(tgt_fold_frames).assign(domain="target_eval"))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    front = ["domain", "metric", "n_folds"]
    rest = [c for c in combined.columns if c not in front]
    return combined[front + rest]
