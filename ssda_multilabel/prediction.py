"""Prediction export as long tables (observed positions only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from ssda_multilabel.model import MultiLabelSSDAModel
from ssda_multilabel.schemas import DrugIndex


def predict_matrix(
    model: MultiLabelSSDAModel,
    x: NDArray[np.float32],
    batch_size: int,
    device: str,
) -> NDArray[np.float32]:
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    preds: list[NDArray[np.float32]] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(dev)
            out = model(batch).cpu().numpy()
            preds.append(out.astype(np.float32))
    return np.vstack(preds) if preds else np.zeros((0, 0), dtype=np.float32)


def build_prediction_long_table(
    scores: NDArray[np.float32],
    y: NDArray[np.float32],
    mask: NDArray[np.float32],
    sample_ids: list[str],
    drug_index: DrugIndex,
    domain: str,
    split_or_role: list[str],
    task_type: str,
    fold: int,
    seed: int,
    cancer_types: dict[str, str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, sid in enumerate(sample_ids):
        for j in range(scores.shape[1]):
            if mask[i, j] <= 0:
                continue
            did = drug_index.index_to_drug[j]
            score = float(scores[i, j])
            clipped = float(np.clip(score, -50.0, 50.0))
            prob = float(1.0 / (1.0 + np.exp(-clipped)))
            if task_type == "classification":
                pred_label = int(prob >= 0.5)
                confidence = prob
            else:
                pred_label = int(prob >= 0.5)
                confidence = score
            row: dict[str, object] = {
                "sample_id": sid,
                "drug_id": did,
                "drug_index": j,
                "domain": domain,
                "split": split_or_role[i] if domain == "source" else "",
                "target_role": split_or_role[i] if domain == "target" else "",
                "ground_truth": float(y[i, j]),
                "mask": 1,
                "pred_score": score,
                "pred_label": pred_label,
                "confidence": confidence,
                "probability": prob,
                "task_type": task_type,
                "fold": fold,
                "seed": seed,
            }
            if cancer_types is not None:
                row["cancer_type"] = cancer_types.get(sid, "Unknown")
            rows.append(row)
    return pd.DataFrame(rows)
