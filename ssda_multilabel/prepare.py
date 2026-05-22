"""Orchestrate multi-label data preparation."""

from __future__ import annotations

import pandas as pd

from ssda_multilabel.cancer_type import load_and_align_cancer_types
from ssda_multilabel.config import MultiLabelConfig
from ssda_multilabel.drug_index import build_drug_index_from_union
from ssda_multilabel.drug_normalize import normalize_drug_name_column
from ssda_multilabel.masks import build_target_nshot_masks
from ssda_multilabel.omics_io import (
    align_omics_features,
    read_omics_table,
    read_response_long,
    resolve_target_omics_sample_id_col,
)
from ssda_multilabel.reports import build_nshot_summary
from ssda_multilabel.response_matrix import long_to_response_matrix
from ssda_multilabel.sample_id import (
    DRUG_NAME_COL,
    SOURCE_OMICS_SAMPLE_ID_COL,
    SOURCE_RESPONSE_SAMPLE_ID_COL,
    TARGET_OMICS_SAMPLE_ID_COL,
    TARGET_RESPONSE_LABEL_COL,
    TARGET_RESPONSE_SAMPLE_ID_COL,
    describe_sample_id_column,
)
from ssda_multilabel.schemas import PreparedData
from ssda_multilabel.split import split_source_samples


def _validate_target_binary(df: pd.DataFrame, col: str) -> None:
    vals = pd.to_numeric(df[col], errors="coerce").dropna().unique()
    bad = set(vals) - {0, 1}
    if bad:
        raise ValueError(f"target response must be binary 0/1, found {bad}")


def _prepare_response_df(path: str) -> pd.DataFrame:
    df = read_response_long(path)
    return normalize_drug_name_column(df)


def prepare_multilabel_data(config: MultiLabelConfig) -> PreparedData:
    tgt_cols = pd.read_csv(config.target_omics_path, nrows=0).columns.tolist()
    target_omics_sid = resolve_target_omics_sample_id_col(tgt_cols)

    src_omics_df = read_omics_table(
        config.source_omics_path,
        SOURCE_OMICS_SAMPLE_ID_COL,
        "source",
    )
    tgt_omics_df = read_omics_table(
        config.target_omics_path,
        target_omics_sid,
        "target",
    )
    source_omics, target_omics, align_report = align_omics_features(src_omics_df, tgt_omics_df)

    src_resp_df = _prepare_response_df(config.source_response_path)
    tgt_resp_df = _prepare_response_df(config.target_response_path)
    _validate_target_binary(tgt_resp_df, TARGET_RESPONSE_LABEL_COL)

    drug_index = build_drug_index_from_union(src_resp_df, tgt_resp_df, DRUG_NAME_COL)
    source_response = long_to_response_matrix(
        src_resp_df,
        list(source_omics.sample_ids),
        drug_index,
        SOURCE_RESPONSE_SAMPLE_ID_COL,
        DRUG_NAME_COL,
        config.source_response_col,
        "source",
        config.duplicate_response_strategy,
        omics_sample_id_col=SOURCE_OMICS_SAMPLE_ID_COL,
        response_sample_id_col=SOURCE_RESPONSE_SAMPLE_ID_COL,
    )
    target_response = long_to_response_matrix(
        tgt_resp_df,
        list(target_omics.sample_ids),
        drug_index,
        TARGET_RESPONSE_SAMPLE_ID_COL,
        DRUG_NAME_COL,
        TARGET_RESPONSE_LABEL_COL,
        "target",
        config.duplicate_response_strategy,
        omics_sample_id_col=target_omics_sid,
        response_sample_id_col=TARGET_RESPONSE_SAMPLE_ID_COL,
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
        SOURCE_OMICS_SAMPLE_ID_COL,
        config.cancer_type_col,
    )
    nshot_summary = build_nshot_summary(target_masks)

    id_report = pd.DataFrame(
        [
            {
                "metric": describe_sample_id_column(
                    list(source_omics.sample_ids)[:200],
                    SOURCE_OMICS_SAMPLE_ID_COL,
                )
            },
            {
                "metric": describe_sample_id_column(
                    list(target_omics.sample_ids)[:200],
                    target_omics_sid,
                )
            },
            {
                "metric": (
                    f"target_join: {TARGET_OMICS_SAMPLE_ID_COL} "
                    f"-> {TARGET_RESPONSE_SAMPLE_ID_COL} via TCGA 3-segment patient key"
                )
            },
            {
                "metric": (
                    f"target_mask_positions={float(target_response.mask.sum())} "
                    f"source_mask_positions={float(source_response.mask.sum())}"
                )
            },
        ]
    )
    alignment_report = pd.concat([align_report, id_report], ignore_index=True)

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
        alignment_report=alignment_report,
        nshot_summary=nshot_summary,
    )
