"""Pretrain data preparation."""

from __future__ import annotations

from codeae_multilabel.contracts import CodeAEMultilabelConfig, PreparedPretrainData
from codeae_multilabel.data.omics import align_omics_features, read_omics_table
from codeae_multilabel.data.subsample import align_paired_omics_for_pretrain
from codeae_multilabel.validators import validate_omics_table
from codeae_multilabel.io import read_csv


def prepare_pretrain_data(config: CodeAEMultilabelConfig) -> PreparedPretrainData:
    src_df = read_csv(config.source_omics_path)
    tgt_df = read_csv(config.target_omics_path)
    validate_omics_table(src_df, config.source_sample_col)
    validate_omics_table(tgt_df, config.target_sample_col)
    source = read_omics_table(config.source_omics_path, config.source_sample_col, "source")
    target = read_omics_table(config.target_omics_path, config.target_sample_col, "target")
    source, target, alignment = align_omics_features(source, target)
    source, target = align_paired_omics_for_pretrain(
        source, target, config.batch_size, config.max_samples, config.seed
    )
    return PreparedPretrainData(source_omics=source, target_omics=target, feature_alignment=alignment)
