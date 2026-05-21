"""Tests for split module."""

from __future__ import annotations

import pandas as pd

from ssda_latent.config import ExperimentConfig
from ssda_latent.data_loading import ExpressionTables
from ssda_latent.split import assign_target_roles, build_split_manifest


def _tiny_tables() -> ExpressionTables:
    n_src = 40
    x_source = pd.DataFrame(
        {f"g{i}": range(n_src) for i in range(5)},
        index=[f"s{i}" for i in range(n_src)],
    )
    y_source = pd.DataFrame(
        {"response": [0] * 20 + [1] * 20},
        index=x_source.index,
    )
    n_tgt = 20
    x_target = pd.DataFrame(
        {f"g{i}": range(n_tgt) for i in range(5)},
        index=[f"t{i}" for i in range(n_tgt)],
    )
    y_target = pd.DataFrame(
        {"response": [0] * 10 + [1] * 10},
        index=x_target.index,
    )
    return ExpressionTables(x_source, y_source, x_target, y_target)


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        drug="test",
        gene="_tp4k",
        n_shot=2,
        random_seed=42,
        source_test_size=0.1,
        n_splits=3,
        encoder="DAE",
        method="adv",
        epochs=1,
        lr=0.001,
        batch_size=8,
        dropout=0.3,
        encoder_h_dims=(8, 4),
        predictor_h_dims=(4, 2),
        device="cpu",
        data_path=".",
        latent_output_dir="save/latent_ssda",
        umap_path="save/figure/",
        result="save/results/sc/",
        sc_all="save/sc/all_path/",
        source_cancer_type_path=None,
        target_cancer_type_path=None,
        sample_id_col="Sample_ID",
        cancer_type_col="Cancer_type",
        missing_cancer_type_policy="unknown",
    )


def test_build_split_manifest_disjoint_folds() -> None:
    tables = _tiny_tables()
    config = _config()
    manifest = build_split_manifest(tables, config)
    assert len(manifest.folds) == 3
    assert len(manifest.source_test_ids) == 4
    all_train_val: set[str] = set()
    for fold in manifest.folds:
        assert fold.train_ids.isdisjoint(fold.val_ids)
        assert manifest.source_test_ids.isdisjoint(fold.train_ids)
        all_train_val |= fold.train_ids | fold.val_ids
    assert len(all_train_val) == 36


def test_target_test_excludes_only_train_labeled() -> None:
    y = pd.DataFrame(
        {"response": [0] * 10 + [1] * 10},
        index=[f"t{i}" for i in range(20)],
    )
    roles = assign_target_roles(y, n_shot=2, random_seed=0)
    from ssda_latent.split import target_test_ids

    test_set = target_test_ids(roles)
    labeled_train = {sid for sid, r in roles.items() if r == "target_labeled_train"}
    assert labeled_train.isdisjoint(test_set)
    assert len(test_set) == len(y) - len(labeled_train)


def test_target_roles_n_shot() -> None:
    # 80/20 split needs >= n_shot per class in both train and val pools
    y = pd.DataFrame(
        {"response": [0] * 10 + [1] * 10},
        index=[f"t{i}" for i in range(20)],
    )
    roles = assign_target_roles(y, n_shot=2, random_seed=0)
    labeled_train = [k for k, v in roles.items() if v == "target_labeled_train"]
    assert len(labeled_train) == 4
