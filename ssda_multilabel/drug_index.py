"""Drug union index: source ∪ target, sorted deterministically."""

from __future__ import annotations

import pandas as pd

from ssda_multilabel.io import write_csv
from ssda_multilabel.schemas import DrugIndex


def build_drug_index_from_union(
    source_response: pd.DataFrame,
    target_response: pd.DataFrame,
    drug_id_col: str,
) -> DrugIndex:
    src_drugs = source_response[drug_id_col].astype(str).unique().tolist()
    tgt_drugs = target_response[drug_id_col].astype(str).unique().tolist()
    union = sorted(set(src_drugs) | set(tgt_drugs))
    if not union:
        raise ValueError("drug list is empty after union of source and target")
    if any(not d.strip() for d in union):
        raise ValueError("empty drug_id found")
    drug_to_index = {d: i for i, d in enumerate(union)}
    index_to_drug = {i: d for d, i in drug_to_index.items()}
    return DrugIndex(
        drug_ids=tuple(union),
        drug_to_index=drug_to_index,
        index_to_drug=index_to_drug,
    )


def save_drug_list(drug_index: DrugIndex, path: str) -> None:
    df = pd.DataFrame(
        {"drug_id": list(drug_index.drug_ids), "drug_index": list(range(drug_index.n_drugs))}
    )
    write_csv(df, path)


def load_drug_list(path: str) -> DrugIndex:
    df = pd.read_csv(path)
    drug_ids = tuple(df["drug_id"].astype(str).tolist())
    drug_to_index = {d: i for i, d in enumerate(drug_ids)}
    index_to_drug = {i: d for d, i in drug_to_index.items()}
    return DrugIndex(
        drug_ids=drug_ids,
        drug_to_index=drug_to_index,
        index_to_drug=index_to_drug,
    )
