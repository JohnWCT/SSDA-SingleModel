"""Orchestrate multi-label data preparation."""

from __future__ import annotations

import pandas as pd

from ssda_multilabel.cancer_type import load_and_align_cancer_types
from ssda_multilabel.config import MultiLabelConfig
from ssda_multilabel.drug_index import build_drug_index_from_union
from ssda_multilabel.masks import build_target_nshot_masks
from ssda_multilabel.omics_io import align_omics_features, read_omics_table, read_response_long
from ssda_multilabel.reports import build_nshot_summary
from ssda_multilabel.response_matrix import long_to_response_matrix
from ssda_multilabel.schemas import PreparedData
from ssda_multilabel.split import split_source_samples


def _validate_target_binary(df: pd.DataFrame, col: str) -> None:
    vals = pd.to_numeric(df[col], errors="coerce").dropna().unique()
    bad = set(vals) - {0, 1}
    if bad:
        raise ValueError(f"target response must be binary 0/1, found {bad}")


def prepare_multilabel_data(config: MultiLabelConfig) -> PreparedData:
    src_omics_df = read_omics_table(config.source_omics_path, config.sample_id_col, "source")
    tgt_omics_df = read_omics_table(config.target_omics_path, config.sample_id_col, "target")
    source_omics, target_omics, align_report = align_omics_features(src_omics_df, tgt_omics_df)

    src_resp_df = read_response_long(config.source_response_path)
    tgt_resp_df = read_response_long(config.target_response_path)
    _validate_target_binary(tgt_resp_df, config.response_col)

    drug_index = build_drug_index_from_union(src_resp_df, tgt_resp_df, config.drug_id_col)
    source_response = long_to_response_matrix(
        src_resp_df,
        list(source_omics.sample_ids),
        drug_index,
        config.sample_id_col,
        config.drug_id_col,
        config.response_col,
        "source",
        config.duplicate_response_strategy,
    )
    target_response = long_to_response_matrix(
        tgt_resp_df,
        list(target_omics.sample_ids),
        drug_index,
        config.sample_id_col,
        config.drug_id_col,
        config.response_col,
        "target",
        config.duplicate_response_strategy,
    )
    target_masks = build_target_nshot_masks(
        target_response.y,
        target_response.mask,
        config.n_shot,
        config.random_seed,
    )
    test_idx, folds = split_source_samples(
        len(source_omics.sample_ids),
        source_response.y,
        source_response.mask,
        config.source_test_size,
        config.n_splits,
        config.random_seed,
    )
    cancer_df, cancer_summary = load_and_align_cancer_types(
        list(source_omics.sample_ids),
        list(target_omics.sample_ids),
        config.source_cancer_type_path,
        config.target_cancer_type_path,
        config.sample_id_col,
        config.cancer_type_col,
    )
    nshot_summary = build_nshot_summary(target_masks)
    return PreparedData(
        source_omics=source_omics,
        target_omics=target_omics,
        source_response=source_response,
        target_response=target_response,
        target_masks=target_masks,
        drug_index=drug_index,
        folds=folds,
        source_test_indices=test_idx,
        cancer_type_df=cancer_df,
        alignment_report=align_report,
        nshot_summary=nshot_summary,
    )
