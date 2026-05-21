"""Tests for prediction metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ssda_latent.prediction import compute_binary_metrics


def test_perfect_auc() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_binary_metrics(y_true, y_score)
    assert m["auc"] == 1.0
    assert m["accuracy"] == 1.0


def test_metrics_for_ids_helper() -> None:
    from ssda_latent.prediction import metrics_for_ids

    y_df = pd.DataFrame({"response": [0, 1]}, index=["a", "b"])
    pred = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "pred_label": [0, 1],
            "probability_class_0": [0.9, 0.1],
            "probability_class_1": [0.1, 0.9],
            "confidence": [0.1, 0.9],
        }
    )
    m = metrics_for_ids(y_df, pred, ["a", "b"])
    assert m["auc"] == 1.0
