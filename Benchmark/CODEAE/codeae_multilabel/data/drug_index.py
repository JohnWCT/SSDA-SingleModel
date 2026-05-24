"""Drug union index."""

from __future__ import annotations

import pandas as pd

from codeae_multilabel.contracts import DrugIndex
from codeae_multilabel.io import write_csv


def _normalize_drug(s: str) -> str:
    return str(s).strip().lower()


def build_drug_index_from_union(
    source_response: pd.DataFrame,
    target_response: pd.DataFrame,
    drug_col: str,
) -> DrugIndex:
    src = {_normalize_drug(d) for d in source_response[drug_col].astype(str)}
    tgt = {_normalize_drug(d) for d in target_response[drug_col].astype(str)}
    union = sorted(src | tgt)
    if not union:
        raise ValueError("drug list is empty after union")
    if any(not d for d in union):
        raise ValueError("empty drug_id found")
    drug_to_index = {d: i for i, d in enumerate(union)}
    index_to_drug = {i: d for d, i in drug_to_index.items()}
    return DrugIndex(drug_ids=tuple(union), drug_to_index=drug_to_index, index_to_drug=index_to_drug)


def save_drug_list(drug_index: DrugIndex, path: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {"drug_id": list(drug_index.drug_ids), "drug_index": list(range(drug_index.n_drugs))}
    )
    write_csv(df, path)
    return df


def load_drug_list(path: str) -> DrugIndex:
    df = pd.read_csv(path)
    drug_ids = tuple(df["drug_id"].astype(str).tolist())
    drug_to_index = {d: i for i, d in enumerate(drug_ids)}
    index_to_drug = {i: d for d, i in drug_to_index.items()}
    return DrugIndex(drug_ids=drug_ids, drug_to_index=drug_to_index, index_to_drug=index_to_drug)
