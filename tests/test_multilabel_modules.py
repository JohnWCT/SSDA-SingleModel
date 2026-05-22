"""Unit tests for ssda_multilabel core modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ssda_multilabel.adaptation import masked_adentropy_loss, masked_entropy_loss
from ssda_multilabel.config import build_arg_parser, config_from_args
from ssda_multilabel.drug_index import build_drug_index_from_union
from ssda_multilabel.losses import masked_bce_with_logits, masked_mse
from ssda_multilabel.masks import build_target_nshot_masks
from ssda_multilabel.model import MultiOutputHead, build_model
from ssda_multilabel.response_matrix import long_to_response_matrix
from ssda_multilabel.seed import set_global_seed
from ssda_multilabel.split import split_source_samples


def test_drug_union_and_order() -> None:
    src = pd.DataFrame({"drug_id": ["B", "A"]})
    tgt = pd.DataFrame({"drug_id": ["C", "A"]})
    idx = build_drug_index_from_union(src, tgt, "drug_id")
    assert idx.drug_ids == ("A", "B", "C")
    assert idx.drug_to_index["B"] == 1


def test_long_to_wide_mask() -> None:
    src = pd.DataFrame({"drug_id": ["D1"]})
    tgt = pd.DataFrame({"drug_id": ["D1"]})
    build_drug_index_from_union(src, tgt, "drug_id")
    resp = pd.DataFrame(
        {
            "Sample_ID": ["S1", "S1"],
            "drug_id": ["D1", "D2"],
            "response": [1.0, 0.0],
        }
    )
    drug_index2 = build_drug_index_from_union(
        pd.DataFrame({"drug_id": ["D1", "D2"]}), pd.DataFrame({"drug_id": []}), "drug_id"
    )
    rm = long_to_response_matrix(
        resp, ["S1"], drug_index2, "Sample_ID", "drug_id", "response", "source", "first"
    )
    assert rm.mask.sum() == 2
    assert rm.y.shape == (1, 2)


def test_target_nshot_reproducible() -> None:
    y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.float32)
    obs = np.ones_like(y)
    m1 = build_target_nshot_masks(y, obs, n_shot=1, seed=7)
    m2 = build_target_nshot_masks(y, obs, n_shot=1, seed=7)
    assert np.allclose(m1.labeled_mask, m2.labeled_mask)


def test_masked_loss_ignores_missing() -> None:
    logits = torch.tensor([[0.0, 0.0]])
    y = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])
    loss = masked_bce_with_logits(logits, y, mask)
    assert loss.ndim == 0
    loss2 = masked_mse(logits, y, mask)
    assert float(loss2.item()) >= 0


def test_model_output_shape() -> None:
    m = build_model(5, 3, "mlp", (8,), dropout=0.0)
    x = torch.randn(4, 5)
    out = m(x)
    z = m.encode(x)
    assert out.shape == (4, 3)
    assert z.shape[0] == 4


def test_adaptation_finite() -> None:
    logits = torch.randn(2, 4)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    assert torch.isfinite(masked_entropy_loss(logits, mask))
    assert torch.isfinite(masked_adentropy_loss(logits, mask))


def test_split_no_leakage() -> None:
    y = np.zeros((10, 2))
    mask = np.ones((10, 2))
    test_idx, folds = split_source_samples(10, y, mask, 0.2, 2, 0)
    for fold in folds:
        tr = set(fold.train_indices)
        va = set(fold.val_indices)
        te = set(fold.test_indices)
        assert tr.isdisjoint(va)
        assert tr.isdisjoint(te)
        assert va.isdisjoint(te)


def test_config_parse() -> None:
    p = build_arg_parser()
    args = p.parse_args(
        [
            "--task_type",
            "classification",
            "--source_omics_path",
            "a.csv",
            "--target_omics_path",
            "b.csv",
            "--source_response_path",
            "c.csv",
            "--target_response_path",
            "d.csv",
        ]
    )
    cfg = config_from_args(args)
    assert cfg.task_type == "classification"
    assert cfg.n_shot == 3


def test_seed_repeatable() -> None:
    set_global_seed(99)
    a = np.random.rand()
    set_global_seed(99)
    b = np.random.rand()
    assert a == b


def test_head_shape() -> None:
    h = MultiOutputHead(8, 5, (16,))
    assert h(torch.randn(2, 8)).shape == (2, 5)
