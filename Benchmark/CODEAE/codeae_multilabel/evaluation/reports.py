"""Human-readable report DataFrames (no file I/O)."""

from __future__ import annotations

import pandas as pd

from codeae_multilabel.contracts import DrugIndex, PreparedFineTuneData, SourceFold


def build_feature_alignment_report(alignment: pd.DataFrame) -> pd.DataFrame:
    return alignment.copy()


def build_drug_availability_report(prepared: PreparedFineTuneData) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for drug_id in prepared.drug_index.drug_ids:
        j = prepared.drug_index.drug_to_index[drug_id]
        src_n = int(prepared.source_response.mask[:, j].sum())
        tgt_n = int(prepared.target_response.mask[:, j].sum())
        rows.append(
            {
                "drug_id": drug_id,
                "drug_index": j,
                "source_observed": src_n,
                "target_observed": tgt_n,
            }
        )
    return pd.DataFrame(rows)


def build_source_split_report(folds: tuple[SourceFold, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in folds:
        for sid in fold.train_sample_ids:
            rows.append({"fold_id": fold.fold_id, "sample_id": sid, "split": "train"})
        for sid in fold.val_sample_ids:
            rows.append({"fold_id": fold.fold_id, "sample_id": sid, "split": "val"})
        for sid in fold.test_sample_ids:
            rows.append({"fold_id": fold.fold_id, "sample_id": sid, "split": "test"})
    return pd.DataFrame(rows)


def build_data_alignment_report(
    source_n: int, target_n: int, n_features: int, drug_index: DrugIndex
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_samples": source_n,
                "target_samples": target_n,
                "n_features": n_features,
                "n_drugs": drug_index.n_drugs,
            }
        ]
    )
