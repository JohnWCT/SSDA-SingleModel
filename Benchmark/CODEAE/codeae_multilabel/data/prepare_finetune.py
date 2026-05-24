"""Fine-tune data preparation."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from codeae_multilabel.contracts import CodeAEMultilabelConfig, OmicsTable, PreparedFineTuneData
from codeae_multilabel.data.sample_id import sample_match_key
from codeae_multilabel.data.cancer_type import (
    infer_dapl_root,
    load_and_align_cancer_types,
    resolve_cancer_type_paths,
)
from codeae_multilabel.data.drug_index import build_drug_index_from_union
from codeae_multilabel.data.omics import align_omics_features, read_omics_table
from codeae_multilabel.data.response_matrix import long_to_response_matrix
from codeae_multilabel.data.split import split_source_samples
from codeae_multilabel.data.subsample import (
    limit_response_long_table,
    subsample_drug_index,
    subsample_omics_table,
)
from codeae_multilabel.io import read_csv
from codeae_multilabel.validators import (
    validate_omics_table,
    validate_response_long_table,
    validate_response_matrix,
)


def prepare_finetune_data(config: CodeAEMultilabelConfig) -> PreparedFineTuneData:
    if not config.source_response_path or not config.target_response_path:
        raise ValueError("fine-tune requires source_response_path and target_response_path")
    src_omics_df = read_csv(config.source_omics_path)
    tgt_omics_df = read_csv(config.target_omics_path)
    validate_omics_table(src_omics_df, config.source_sample_col)
    validate_omics_table(tgt_omics_df, config.target_sample_col)
    source = read_omics_table(config.source_omics_path, config.source_sample_col, "source")
    target = read_omics_table(config.target_omics_path, config.target_sample_col, "target")
    source, target, alignment = align_omics_features(source, target)
    src_resp_df = read_csv(config.source_response_path)
    tgt_resp_df = read_csv(config.target_response_path)
    resp_keys = {
        sample_match_key(s, column_hint=config.target_response_sample_col)
        for s in tgt_resp_df[config.target_response_sample_col].astype(str)
    }
    keep = [
        sid
        for sid in target.sample_ids
        if sample_match_key(sid, column_hint=config.target_sample_col) in resp_keys
    ]
    if keep:
        if len(keep) < len(target.sample_ids):
            target = OmicsTable(
                x=target.x.loc[keep],
                sample_ids=tuple(keep),
                feature_names=target.feature_names,
                domain="target",
            )
    else:
        raise ValueError("no target omics samples join to target response after ID normalization")
    source = subsample_omics_table(source, config.max_samples, config.seed)
    target = subsample_omics_table(target, config.max_samples, config.seed + 1)
    validate_response_long_table(
        src_resp_df,
        config.source_sample_col,
        config.drug_col,
        config.source_response_col,
        config.task_type,
        "source",
    )
    validate_response_long_table(
        tgt_resp_df,
        config.target_response_sample_col,
        config.drug_col,
        config.target_response_col,
        config.task_type,
        "target",
    )
    drug_index_full = build_drug_index_from_union(src_resp_df, tgt_resp_df, config.drug_col)
    drug_index = subsample_drug_index(drug_index_full, config.max_drugs)
    allowed = set(drug_index.drug_ids)
    src_resp_df = limit_response_long_table(src_resp_df, config.drug_col, allowed)
    tgt_resp_df = limit_response_long_table(tgt_resp_df, config.drug_col, allowed)
    src_sem: Literal["binary", "continuous"] = (
        "binary" if config.task_type == "classification" else "continuous"
    )
    tgt_sem: Literal["binary", "continuous"] = (
        "binary" if config.task_type == "classification" else "continuous"
    )
    source_response = long_to_response_matrix(
        src_resp_df,
        list(source.sample_ids),
        drug_index,
        config.source_sample_col,
        config.drug_col,
        config.source_response_col,
        "source",
        src_sem,
        config.duplicate_strategy,
        omics_sample_id_col=config.source_sample_col,
        response_sample_id_col=config.source_sample_col,
    )
    target_response = long_to_response_matrix(
        tgt_resp_df,
        list(target.sample_ids),
        drug_index,
        config.target_response_sample_col,
        config.drug_col,
        config.target_response_col,
        "target",
        tgt_sem,
        config.duplicate_strategy,
        omics_sample_id_col=config.target_sample_col,
        response_sample_id_col=config.target_response_sample_col,
    )
    validate_response_matrix(source_response)
    validate_response_matrix(target_response)
    folds = split_source_samples(
        list(source.sample_ids),
        source_response.y,
        source_response.mask,
        config.source_test_size,
        config.n_splits,
        config.seed,
    )
    src_ct_path = config.source_cancer_type_path
    tgt_ct_path = config.target_cancer_type_path
    if src_ct_path is None or tgt_ct_path is None:
        auto_src, auto_tgt = resolve_cancer_type_paths(
            config.source_omics_path,
            config.target_omics_path,
            infer_dapl_root(config.source_omics_path, config.target_omics_path),
        )
        src_ct_path = src_ct_path or auto_src
        tgt_ct_path = tgt_ct_path or auto_tgt
    cancer_type_table, cancer_type_summary = load_and_align_cancer_types(
        list(source.sample_ids),
        list(target.sample_ids),
        src_ct_path,
        tgt_ct_path,
        config.target_sample_col,
    )
    return PreparedFineTuneData(
        source_omics=source,
        target_omics=target,
        source_response=source_response,
        target_response=target_response,
        drug_index=drug_index,
        drug_index_full=drug_index_full,
        folds=tuple(folds),
        cancer_type_table=cancer_type_table,
        cancer_type_summary=cancer_type_summary,
        feature_alignment=alignment,
    )
