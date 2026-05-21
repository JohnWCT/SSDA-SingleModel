"""Tests for paths module."""

from __future__ import annotations

from ssda_latent.config import ExperimentConfig
from ssda_latent.paths import RunLayout


def test_run_layout_paths() -> None:
    cfg = ExperimentConfig(
        drug="Gefitinib",
        gene="_tp4k",
        n_shot=3,
        random_seed=42,
        source_test_size=0.1,
        n_splits=5,
        encoder="DAE",
        method="adv",
        epochs=1,
        lr=0.001,
        batch_size=32,
        dropout=0.3,
        encoder_h_dims=(512, 256),
        predictor_h_dims=(64, 32),
        device="cpu",
        data_path=".",
        output_dir="outputs",
        latent_output_dir="outputs/latent_ssda",
        umap_path="outputs/legacy/figure/",
        result="outputs/legacy/results/sc/",
        sc_all="outputs/legacy/sc/all_path/",
        source_cancer_type_path=None,
        target_cancer_type_path=None,
        sample_id_col="Sample_ID",
        cancer_type_col="Cancer_type",
        missing_cancer_type_policy="unknown",
    )
    layout = RunLayout.from_config(cfg)
    assert layout.run_dir.name == "seed_42"
    assert layout.fold_dir(2).name == "fold_2"
