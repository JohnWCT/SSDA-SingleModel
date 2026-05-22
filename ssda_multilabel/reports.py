"""Human-readable preparation reports."""

from __future__ import annotations

import pandas as pd

from ssda_multilabel.schemas import DrugIndex, TargetMasks


def build_nshot_summary(target_masks: TargetMasks) -> pd.DataFrame:
    rows = [{"warning": w} for w in target_masks.warnings]
    if not rows:
        rows = [{"warning": "none"}]
    labeled_per_drug = target_masks.labeled_mask.sum(axis=0)
    for j, n in enumerate(labeled_per_drug):
        rows.append({"warning": f"drug_col_{j}_labeled_count", "value": str(float(n))})
    return pd.DataFrame(rows)


def build_missing_data_report(
    source_mask_sum: float,
    target_mask_sum: float,
    n_drugs: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "source_observed_positions", "value": source_mask_sum},
            {"metric": "target_observed_positions", "value": target_mask_sum},
            {"metric": "n_drugs", "value": n_drugs},
        ]
    )


def drug_list_report(drug_index: DrugIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {"drug_id": list(drug_index.drug_ids), "drug_index": range(drug_index.n_drugs)}
    )
