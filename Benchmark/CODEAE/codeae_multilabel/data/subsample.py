"""Optional subsampling for smoke / debug runs (does not change default full-run behavior)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from codeae_multilabel.contracts import DrugIndex, OmicsTable


def subsample_omics_table(table: OmicsTable, max_samples: int | None, seed: int) -> OmicsTable:
    if max_samples is None or len(table.sample_ids) <= max_samples:
        return table
    return subsample_omics_to_count(table, max_samples, seed)


def subsample_omics_to_count(table: OmicsTable, n: int, seed: int) -> OmicsTable:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(table.sample_ids) <= n:
        return table
    rng = np.random.default_rng(seed)
    pick = sorted(int(i) for i in rng.choice(len(table.sample_ids), size=n, replace=False))
    sids = tuple(table.sample_ids[i] for i in pick)
    x = table.x.loc[list(sids)]
    return OmicsTable(
        x=x,
        sample_ids=sids,
        feature_names=table.feature_names,
        domain=table.domain,
    )


def align_paired_omics_for_pretrain(
    source: OmicsTable,
    target: OmicsTable,
    batch_size: int,
    max_samples: int | None,
    seed: int,
) -> tuple[OmicsTable, OmicsTable]:
    """Match source/target counts and use a multiple of batch_size (CODE-AE WGAN pairing)."""
    cap = min(len(source.sample_ids), len(target.sample_ids))
    if max_samples is not None:
        cap = min(cap, max_samples)
    n = max((cap // batch_size) * batch_size, batch_size)
    n = min(n, cap)
    return (
        subsample_omics_to_count(source, n, seed),
        subsample_omics_to_count(target, n, seed + 1),
    )


def limit_response_long_table(
    df: pd.DataFrame,
    drug_col: str,
    allowed_drugs: set[str],
) -> pd.DataFrame:
    from codeae_multilabel.data.drug_index import _normalize_drug

    drugs = df[drug_col].astype(str).map(_normalize_drug)
    return df.loc[drugs.isin(allowed_drugs)].copy()


def subsample_drug_index(drug_index: DrugIndex, max_drugs: int | None) -> DrugIndex:
    if max_drugs is None or drug_index.n_drugs <= max_drugs:
        return drug_index
    kept = drug_index.drug_ids[:max_drugs]
    drug_to_index = {d: i for i, d in enumerate(kept)}
    index_to_drug = {i: d for d, i in drug_to_index.items()}
    return DrugIndex(drug_ids=tuple(kept), drug_to_index=drug_to_index, index_to_drug=index_to_drug)
