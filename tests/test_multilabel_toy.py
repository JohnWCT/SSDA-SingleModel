"""End-to-end toy dataset test for ssda_multilabel (CPU, lightweight)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ssda_multilabel.config import MultiLabelConfig
from ssda_multilabel.export import ArtifactWriter
from ssda_multilabel.latent import encode_latent_dict
from ssda_multilabel.model import build_model
from ssda_multilabel.prediction import predict_matrix
from ssda_multilabel.prepare import prepare_multilabel_data
from ssda_multilabel.seed import set_global_seed
from ssda_multilabel.training import MultiLabelSSDTrainer


def _write_toy(tmp_path: Path) -> MultiLabelConfig:
    rng = np.random.default_rng(0)
    n_src, n_tgt, n_feat = 8, 6, 10
    src_ids = [f"S{i}" for i in range(n_src)]
    tgt_ids = [f"T{i}" for i in range(n_tgt)]
    feats = [f"g{j}" for j in range(n_feat)]

    def omics(ids: list[str], path: Path) -> None:
        rows = {"Sample_ID": ids}
        for f in feats:
            rows[f] = rng.normal(size=len(ids)).tolist()
        pd.DataFrame(rows).to_csv(path, index=False)

    omics(src_ids, tmp_path / "source_omics.csv")
    omics(tgt_ids, tmp_path / "target_omics.csv")

    src_rows = []
    for sid in src_ids:
        for d in ["DrugShared", "DrugSourceOnly"]:
            if d == "DrugSparse" and sid == "S0":
                continue
            src_rows.append({"Sample_ID": sid, "drug_id": d, "response": int(rng.integers(0, 2))})
    pd.DataFrame(src_rows).to_csv(tmp_path / "source_response.csv", index=False)

    tgt_rows = []
    for sid in tgt_ids:
        for d in ["DrugShared", "DrugTargetOnly"]:
            tgt_rows.append({"Sample_ID": sid, "drug_id": d, "response": int(rng.integers(0, 2))})
    # class imbalance for n-shot warning
    tgt_rows.append({"Sample_ID": "T0", "drug_id": "DrugShared", "response": 1})
    pd.DataFrame(tgt_rows).to_csv(tmp_path / "target_response.csv", index=False)

    pd.DataFrame({"sample_id": src_ids, "cancer_type": ["LUAD"] * n_src}).to_csv(
        tmp_path / "source_cancer.csv", index=False
    )
    pd.DataFrame({"sample_id": tgt_ids, "cancer_type": ["LUSC"] * n_tgt}).to_csv(
        tmp_path / "target_cancer.csv", index=False
    )

    return MultiLabelConfig(
        task_type="classification",
        source_omics_path=str(tmp_path / "source_omics.csv"),
        target_omics_path=str(tmp_path / "target_omics.csv"),
        source_response_path=str(tmp_path / "source_response.csv"),
        target_response_path=str(tmp_path / "target_response.csv"),
        sample_id_col="Sample_ID",
        drug_id_col="drug_id",
        response_col="response",
        source_cancer_type_path=str(tmp_path / "source_cancer.csv"),
        target_cancer_type_path=str(tmp_path / "target_cancer.csv"),
        cancer_type_col="cancer_type",
        random_seed=42,
        source_test_size=0.25,
        n_splits=2,
        n_shot=3,
        reg_loss="mse",
        lambda_adapt=0.1,
        latent_output_dir=str(tmp_path / "save"),
        encoder="mlp",
        encoder_h_dims=(16,),
        epochs=1,
        lr=1e-2,
        batch_size=4,
        dropout=0.1,
        device="cpu",
        duplicate_response_strategy="first",
        exclude_unknown_cancer_type_for_kmeans=False,
    )


def test_toy_pipeline_end_to_end(tmp_path: Path) -> None:
    config = _write_toy(tmp_path)
    set_global_seed(config.random_seed)
    prepared = prepare_multilabel_data(config)
    assert prepared.drug_index.n_drugs == 3
    assert "DrugSourceOnly" in prepared.drug_index.drug_ids
    assert "DrugTargetOnly" in prepared.drug_index.drug_ids
    assert prepared.target_masks.labeled_mask.sum() >= 0
    assert np.allclose(
        prepared.target_masks.labeled_mask + prepared.target_masks.unlabeled_mask,
        prepared.target_masks.observed_mask,
    )

    fold = prepared.folds[0]
    model = build_model(
        prepared.source_omics.x.shape[1],
        prepared.drug_index.n_drugs,
        config.encoder,
        config.encoder_h_dims,
        dropout=config.dropout,
    )
    trainer = MultiLabelSSDTrainer(model, config)
    result = trainer.train_fold(prepared, fold)
    assert len(result.epoch_logs) == 1

    scores = predict_matrix(model, prepared.source_omics.x, 4, "cpu")
    assert scores.shape == (len(prepared.source_omics.sample_ids), prepared.drug_index.n_drugs)

    latent = encode_latent_dict(
        model, prepared.source_omics.x, list(prepared.source_omics.sample_ids), 4, "cpu"
    )
    assert len(latent) == len(prepared.source_omics.sample_ids)

    writer = ArtifactWriter(config.latent_output_dir, config.random_seed)
    writer.write_preparation_artifacts(
        prepared.drug_index,
        prepared.source_response,
        prepared.target_response,
        prepared.target_masks,
        prepared.alignment_report,
        prepared.nshot_summary,
        pd.DataFrame(),
    )
    assert (writer.root / "drug_list.csv").exists()
