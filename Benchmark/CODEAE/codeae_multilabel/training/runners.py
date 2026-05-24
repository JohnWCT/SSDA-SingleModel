"""Pretrain and fine-tune orchestration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from codeae_multilabel.contracts import CodeAEMultilabelConfig, TrainingResult
from codeae_multilabel.data.prepare_finetune import prepare_finetune_data
from codeae_multilabel.data.prepare_pretrain import prepare_pretrain_data
from codeae_multilabel.evaluation.fold_summary import (
    aggregate_per_drug_metrics,
    aggregate_scalar_metrics,
    aggregate_summary_metrics,
    build_combined_eval_summary,
    filter_source_test,
    filter_target_eval,
    normalize_per_drug_columns,
    summary_to_ssda_wide,
)
from codeae_multilabel.evaluation.metrics import compute_metrics_from_predictions
from codeae_multilabel.evaluation.reports import (
    build_data_alignment_report,
    build_drug_availability_report,
    build_feature_alignment_report,
    build_source_split_report,
)
from codeae_multilabel.export.artifacts import ArtifactWriter
from codeae_multilabel.export.latent import extract_latent_table, latent_table_to_dict
from codeae_multilabel.export.latent_eval import (
    build_cancer_map,
    compute_distribution_metrics,
    compute_kmeans_cancer_type_metrics,
    plot_tsne_cancer_type,
    plot_tsne_domain,
)
from codeae_multilabel.model.checkpoint import load_finetune_checkpoint, load_pretrain_checkpoint
from codeae_multilabel.model.heads import MultiOutputDrugHead
from codeae_multilabel.model.legacy_adapter import build_legacy_codeae_components, run_pretrain
from codeae_multilabel.model.wrapper import MultiLabelCodeAEModel
from codeae_multilabel.seed import set_global_seed
from codeae_multilabel.training.trainer import build_trainer


def _build_multilabel_model(config: CodeAEMultilabelConfig, n_features: int, n_drugs: int) -> MultiLabelCodeAEModel:
    components = build_legacy_codeae_components(config, n_features)
    head = MultiOutputDrugHead(
        input_dim=config.latent_dim,
        hidden_dims=list(config.classifier_hidden_dims),
        n_drugs=n_drugs,
        dropout=config.dop,
    )
    return MultiLabelCodeAEModel(components["shared_encoder"], head)


def _write_cross_fold_summaries(
    writer: ArtifactWriter,
    all_src_summary: list[pd.DataFrame],
    all_tgt_summary: list[pd.DataFrame],
    all_src_per_drug: list[pd.DataFrame],
    all_tgt_per_drug: list[pd.DataFrame],
    all_latent_metrics: list[pd.DataFrame],
    all_kmeans: list[pd.DataFrame],
) -> None:
    if all_src_summary:
        writer.write_summary(
            "source_test_metrics_summary_across_folds.csv",
            pd.concat(all_src_summary, ignore_index=True),
        )
        writer.write_summary(
            "source_test_metrics_summary_fold_mean_std.csv",
            aggregate_summary_metrics(all_src_summary),
        )
    if all_tgt_summary:
        writer.write_summary(
            "target_eval_metrics_summary_across_folds.csv",
            pd.concat(all_tgt_summary, ignore_index=True),
        )
        writer.write_summary(
            "target_eval_metrics_summary_fold_mean_std.csv",
            aggregate_summary_metrics(all_tgt_summary),
        )
    if all_src_summary or all_tgt_summary:
        writer.write_summary(
            "eval_metrics_summary_fold_mean_std.csv",
            build_combined_eval_summary(all_src_summary, all_tgt_summary),
        )
    if all_src_per_drug:
        writer.write_summary(
            "source_test_metrics_per_drug_fold_mean_std.csv",
            aggregate_per_drug_metrics(all_src_per_drug),
        )
    if all_tgt_per_drug:
        writer.write_summary(
            "target_eval_metrics_per_drug_fold_mean_std.csv",
            aggregate_per_drug_metrics(all_tgt_per_drug),
        )
    if all_latent_metrics:
        writer.write_summary(
            "latent_metrics_summary.csv",
            pd.concat(all_latent_metrics, ignore_index=True),
        )
    if all_kmeans:
        writer.write_summary(
            "kmeans_cancer_type_summary.csv",
            pd.concat(all_kmeans, ignore_index=True),
        )
        writer.write_summary(
            "kmeans_cancer_type_fold_mean_std.csv",
            aggregate_scalar_metrics(all_kmeans),
        )


class PretrainRunner:
    def __init__(self, config: CodeAEMultilabelConfig) -> None:
        self.config = config

    def run(self) -> None:
        set_global_seed(self.config.seed)
        writer = ArtifactWriter(self.config.output_dir, overwrite=self.config.overwrite)
        writer.write_config(self.config)
        prepared = prepare_pretrain_data(self.config)
        writer.write_feature_alignment(build_feature_alignment_report(prepared.feature_alignment))
        pre_dir = Path(self.config.output_dir) / "pretrain"
        pre_dir.mkdir(parents=True, exist_ok=True)
        shared_encoder = run_pretrain(self.config, prepared, str(pre_dir))
        writer.write_pretrain_checkpoint(shared_encoder.state_dict())
        writer.write_run_manifest()


class FineTuneRunner:
    def __init__(self, config: CodeAEMultilabelConfig) -> None:
        self.config = config
        if not config.pretrain_checkpoint:
            raise ValueError("fine-tune requires pretrain_checkpoint")

    def run(self) -> None:
        set_global_seed(self.config.seed)
        writer = ArtifactWriter(self.config.output_dir, overwrite=self.config.overwrite)
        writer.write_config(self.config)
        prepared = prepare_finetune_data(self.config)
        writer.write_drug_list(prepared.drug_index_full)
        writer.write_feature_alignment(build_feature_alignment_report(prepared.feature_alignment))
        if prepared.cancer_type_summary is not None:
            writer.write_summary("cancer_type_summary.csv", prepared.cancer_type_summary)
        writer.write_data_alignment(
            build_data_alignment_report(
                len(prepared.source_omics.sample_ids),
                len(prepared.target_omics.sample_ids),
                len(prepared.source_omics.feature_names),
                prepared.drug_index,
            )
        )
        writer.write_drug_availability(build_drug_availability_report(prepared))
        n_features = len(prepared.source_omics.feature_names)
        n_drugs = prepared.drug_index.n_drugs
        cancer_map = build_cancer_map(prepared.cancer_type_table)

        fold_summaries: list[dict[str, object]] = []
        all_src_summary: list[pd.DataFrame] = []
        all_tgt_summary: list[pd.DataFrame] = []
        all_src_per_drug: list[pd.DataFrame] = []
        all_tgt_per_drug: list[pd.DataFrame] = []
        all_latent_metrics: list[pd.DataFrame] = []
        all_kmeans: list[pd.DataFrame] = []

        for fold in prepared.folds:
            fold_dir = writer.fold_dir(fold.fold_id)
            model = _build_multilabel_model(self.config, n_features, n_drugs)
            model.to(self.config.device)
            ckpt_path = self.config.pretrain_checkpoint
            if ckpt_path is None:
                raise ValueError("fine-tune requires pretrain_checkpoint")
            load_report = load_pretrain_checkpoint(model, ckpt_path)
            writer.write_checkpoint_load_report(fold.fold_id, load_report)
            trainer = build_trainer(model, self.config)
            result, bundle = trainer.train_fold(fold, prepared, str(fold_dir))
            load_finetune_checkpoint(model, result.best_model_path)
            model.eval()
            writer.write_fold_training_result(fold.fold_id, result)
            writer.write_fold_predictions(fold.fold_id, bundle)

            src_eval = filter_source_test(bundle.source_predictions)
            tgt_eval = filter_target_eval(bundle.target_predictions)
            src_per, src_sum_long = compute_metrics_from_predictions(
                src_eval, self.config.task_type, "source"
            )
            tgt_per, tgt_sum_long = compute_metrics_from_predictions(
                tgt_eval, self.config.task_type, "target"
            )
            src_sum = summary_to_ssda_wide(src_sum_long)
            tgt_sum = summary_to_ssda_wide(tgt_sum_long)
            src_per = normalize_per_drug_columns(src_per)
            tgt_per = normalize_per_drug_columns(tgt_per)
            writer.write_fold_csv(fold.fold_id, "source_metrics_per_drug.csv", src_per)
            writer.write_fold_csv(fold.fold_id, "source_metrics_summary.csv", src_sum)
            writer.write_fold_csv(fold.fold_id, "target_metrics_per_drug.csv", tgt_per)
            writer.write_fold_csv(fold.fold_id, "target_metrics_summary.csv", tgt_sum)

            all_src_per_drug.append(src_per.assign(fold=fold.fold_id))
            all_src_summary.append(src_sum.assign(fold=fold.fold_id))
            all_tgt_per_drug.append(tgt_per.assign(fold=fold.fold_id))
            all_tgt_summary.append(tgt_sum.assign(fold=fold.fold_id))

            src_latent = extract_latent_table(
                model, prepared.source_omics, self.config.batch_size, self.config.device
            )
            tgt_latent = extract_latent_table(
                model, prepared.target_omics, self.config.batch_size, self.config.device
            )
            writer.write_latent(fold.fold_id, "source", src_latent)
            writer.write_latent(fold.fold_id, "target", tgt_latent)

            src_latent_dict = latent_table_to_dict(src_latent)
            tgt_latent_dict = latent_table_to_dict(tgt_latent)
            writer.write_combined_latent_pkl(fold.fold_id, src_latent_dict, tgt_latent_dict)
            latent_m = compute_distribution_metrics(src_latent_dict, tgt_latent_dict)
            writer.write_fold_csv(
                fold.fold_id,
                "latent_distribution_metrics.csv",
                pd.DataFrame([latent_m]),
            )
            all_latent_metrics.append(pd.DataFrame([latent_m]).assign(fold=fold.fold_id))

            plot_tsne_domain(
                src_latent_dict,
                tgt_latent_dict,
                self.config.seed,
                fold_dir / "tsne_domain_mixing.png",
            )
            if cancer_map:
                combined_ids = (
                    list(src_latent["sample_id"].astype(str))
                    + list(tgt_latent["sample_id"].astype(str))
                )
                cancer_labels = [cancer_map.get(sid, "Unknown") for sid in combined_ids]
                plot_tsne_cancer_type(
                    src_latent_dict,
                    tgt_latent_dict,
                    cancer_labels,
                    combined_ids,
                    self.config.seed,
                    fold_dir / "tsne_cancer_type.png",
                )

            combined_latent = {**src_latent_dict, **tgt_latent_dict}
            kmeans_m = compute_kmeans_cancer_type_metrics(
                combined_latent, cancer_map, self.config.seed
            )
            writer.write_fold_csv(
                fold.fold_id,
                "kmeans_cancer_type_metrics.csv",
                pd.DataFrame([kmeans_m]),
            )
            all_kmeans.append(pd.DataFrame([kmeans_m]).assign(fold=fold.fold_id))

            fold_summaries.append(
                {
                    "fold_id": fold.fold_id,
                    "best_epoch": result.best_epoch,
                    "best_metric_name": result.best_metric_name,
                    "best_metric_value": result.best_metric_value,
                    "seed": self.config.seed,
                    "task_type": self.config.task_type,
                    "n_drugs": n_drugs,
                }
            )

        writer.write_fold_summary(pd.DataFrame(fold_summaries))
        _write_cross_fold_summaries(
            writer,
            all_src_summary,
            all_tgt_summary,
            all_src_per_drug,
            all_tgt_per_drug,
            all_latent_metrics,
            all_kmeans,
        )
        split_report = build_source_split_report(prepared.folds)
        from codeae_multilabel.io import write_csv

        write_csv(split_report, str(Path(self.config.output_dir) / "source_split.csv"))
        writer.write_run_manifest()
