"""Full-sample prediction and classification metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import utils
from ssda_latent.cancer_type import cancer_type_label
from ssda_latent.config import ExperimentConfig
from ssda_latent.latent import get_encoder_latent
from ssda_latent.split import SampleSplit, SplitManifest, source_split_for_fold


def predict_dataframe(
    encoder: nn.Module,
    predictor: nn.Module,
    adentropy_p: nn.Module,
    x_df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    encoder.eval()
    predictor.eval()
    adentropy_p.eval()
    sample_ids = x_df.index.astype(str).tolist()
    x_tensor = torch.FloatTensor(x_df.values)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=False)
    probs_list: list[NDArray[np.float64]] = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            feature = get_encoder_latent(encoder, batch_x, deterministic=True)
            hidden = predictor(feature)
            logits = adentropy_p(hidden)
            probs = nn.Softmax(dim=1)(logits).cpu().numpy()
            probs_list.append(probs)
    probs_all = np.vstack(probs_list)
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "pred_label": probs_all.argmax(axis=1).astype(int),
            "probability_class_0": probs_all[:, 0],
            "probability_class_1": probs_all[:, 1],
            "confidence": probs_all[:, 1],
        }
    )


def compute_binary_metrics(
    y_true: NDArray[np.int_], y_score: NDArray[np.float64]
) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "auc": float(utils.roc_auc_score_trainval(y_true, y_score)),
        "aupr": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def build_prediction_table(
    pred_df: pd.DataFrame,
    samples: list[SampleSplit],
    config: ExperimentConfig,
    fold_index: int,
    domain: str,
    cancer_map: dict[str, str] | None,
    manifest: SplitManifest | None,
) -> pd.DataFrame:
    meta = {s.sample_id: s for s in samples if s.domain == domain}
    rows = []
    for _, row in pred_df.iterrows():
        sid = str(row["sample_id"])
        sp = meta.get(sid)
        if sp is None:
            continue
        split_col = sp.source_split
        if domain == "source" and manifest is not None:
            split_col = source_split_for_fold(manifest, fold_index, sid)
        rows.append(
            {
                "sample_id": sid,
                "domain": domain,
                "split": split_col if domain == "source" else None,
                "target_role": sp.target_role if domain == "target" else None,
                "response_label": sp.response_label,
                "pred_label": int(row["pred_label"]),
                "confidence": float(row["confidence"]),
                "probability_class_0": float(row["probability_class_0"]),
                "probability_class_1": float(row["probability_class_1"]),
                "fold": fold_index,
                "seed": config.random_seed,
                "drug": config.drug,
                "cancer_type": cancer_type_label(
                    sid, cancer_map or {}, config.missing_cancer_type_policy
                ),
            }
        )
    return pd.DataFrame(rows)


def metrics_for_ids(
    y_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    sample_ids: list[str],
) -> dict[str, float]:
    ids = [str(i) for i in sample_ids]
    y_true = y_df.reindex(ids)["response"].to_numpy(dtype=np.int_)
    merged = pred_df.set_index("sample_id").reindex(ids)
    y_score = merged["probability_class_1"].to_numpy(dtype=np.float64)
    return compute_binary_metrics(y_true, y_score)
