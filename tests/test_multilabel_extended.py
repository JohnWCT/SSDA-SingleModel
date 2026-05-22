"""Extended coverage tests for ssda_multilabel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ssda_multilabel.cancer_type import load_and_align_cancer_types
from ssda_multilabel.drug_index import build_drug_index_from_union, load_drug_list, save_drug_list
from ssda_multilabel.io import ensure_dir, read_pickle, write_pickle
from ssda_multilabel.losses import masked_huber, masked_mae, masked_regression_loss
from ssda_multilabel.metrics import (
    compute_classification_metrics_per_drug,
    compute_classification_metrics_summary,
    compute_metrics_from_predictions,
    compute_regression_metrics_per_drug,
)
from ssda_multilabel.model import build_model
from ssda_multilabel.prediction import build_prediction_long_table
from ssda_multilabel.reports import build_missing_data_report


def test_losses_all_types() -> None:
    pred = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0]])
    assert masked_mae(pred, y, mask).ndim == 0
    assert masked_huber(pred, y, mask).ndim == 0
    assert masked_regression_loss(pred, y, mask, "mse").ndim == 0
    assert masked_regression_loss(pred, y, mask, "mae").ndim == 0
    assert masked_regression_loss(pred, y, mask, "huber").ndim == 0


def test_metrics_classification_and_regression() -> None:
    df = pd.DataFrame(
        {
            "drug_id": ["D1", "D1", "D2", "D2"],
            "ground_truth": [0, 1, 1, 0],
            "probability": [0.2, 0.8, 0.9, 0.1],
            "pred_label": [0, 1, 1, 0],
            "pred_score": [0.2, 0.8, 1.2, 0.5],
            "domain": ["source"] * 4,
        }
    )
    per = compute_classification_metrics_per_drug(df)
    summ = compute_classification_metrics_summary(per)
    assert len(per) == 2
    assert "macro" in summ.columns
    reg_df = df.copy()
    reg_per = compute_regression_metrics_per_drug(reg_df)
    assert "rmse" in reg_per.columns
    _, _ = compute_metrics_from_predictions(df, "classification", "source")


def test_prediction_long_table() -> None:
    idx = build_drug_index_from_union(
        pd.DataFrame({"drug_id": ["A"]}), pd.DataFrame({"drug_id": ["A"]}), "drug_id"
    )
    scores = np.array([[0.5]], dtype=np.float32)
    y = np.array([[1.0]], dtype=np.float32)
    mask = np.array([[1.0]], dtype=np.float32)
    table = build_prediction_long_table(
        scores,
        y,
        mask,
        ["S1"],
        idx,
        "source",
        ["source_test"],
        "classification",
        0,
        42,
    )
    assert len(table) == 1
    assert table.loc[0, "drug_id"] == "A"


def test_drug_list_roundtrip(tmp_path: Path) -> None:
    idx = build_drug_index_from_union(
        pd.DataFrame({"drug_id": ["Z", "A"]}),
        pd.DataFrame({"drug_id": ["B"]}),
        "drug_id",
    )
    path = tmp_path / "drug_list.csv"
    save_drug_list(idx, str(path))
    loaded = load_drug_list(str(path))
    assert loaded.drug_ids == ("A", "B", "Z")


def test_cancer_type_unknown() -> None:
    full, summary = load_and_align_cancer_types(
        ["S1"], ["T1"], None, None, "sample_id", "cancer_type"
    )
    assert full.loc[0, "cancer_type"] == "Unknown"
    assert len(summary) == 2


def test_export_and_io(tmp_path: Path) -> None:
    ensure_dir(tmp_path / "sub")
    write_pickle({"a": 1}, tmp_path / "x.pkl")
    assert read_pickle(tmp_path / "x.pkl") == {"a": 1}
    rep = build_missing_data_report(10.0, 5.0, 3)
    assert len(rep) == 3


def test_dae_model_build() -> None:
    m = build_model(6, 2, "dae", (8,), dropout=0.1)
    x = torch.randn(3, 6)
    assert m(x).shape == (3, 2)
