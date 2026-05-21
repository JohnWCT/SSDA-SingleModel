"""Per-fold export: latent, predictions, metrics, plots."""

from __future__ import annotations

from ssda_latent.artifacts import ArtifactWriter
from ssda_latent.cancer_type import CancerTypeRegistry, cancer_type_label
from ssda_latent.config import ExperimentConfig
from ssda_latent.data_loading import ExpressionTables
from ssda_latent.latent import encode_latent_dict
from ssda_latent.latent_eval import (
    compute_distribution_metrics,
    compute_kmeans_cancer_type_metrics,
    plot_tsne_cancer_type,
    plot_tsne_domain,
)
from ssda_latent.paths import RunLayout
from ssda_latent.prediction import (
    build_prediction_table,
    metrics_for_ids,
    predict_dataframe,
)
from ssda_latent.split import (
    SplitManifest,
    get_source_ids_for_fold,
    source_split_for_fold,
)
from ssda_latent.training_adapter import TrainingResult, resolve_device


class ExportPipeline:
    def __init__(
        self,
        config: ExperimentConfig,
        registry: CancerTypeRegistry,
        writer: ArtifactWriter,
    ) -> None:
        self.config = config
        self.registry = registry
        self.writer = writer

    def run(
        self,
        fold_index: int,
        training: TrainingResult,
        tables: ExpressionTables,
        manifest: SplitManifest,
        layout: RunLayout,
    ) -> None:
        device = resolve_device(self.config.device)
        encoder = training.encoder
        predictor = training.predictor
        adentropy = training.adentropy_p

        source_latent = encode_latent_dict(encoder, tables.x_source, device, self.config.batch_size)
        target_latent = encode_latent_dict(encoder, tables.x_target, device, self.config.batch_size)

        src_pred_raw = predict_dataframe(
            encoder, predictor, adentropy, tables.x_source, device, self.config.batch_size
        )
        tgt_pred_raw = predict_dataframe(
            encoder, predictor, adentropy, tables.x_target, device, self.config.batch_size
        )

        samples = manifest.all_samples(tables)
        policy = self.config.missing_cancer_type_policy
        cancer_src = self.registry.source_map if self.registry.is_available else {}
        cancer_tgt = self.registry.target_map if self.registry.is_available else {}
        cancer_all = self.registry.combined_map() if self.registry.is_available else {}

        src_pred = build_prediction_table(
            src_pred_raw, samples, self.config, fold_index, "source", cancer_src, manifest
        )
        tgt_pred = build_prediction_table(
            tgt_pred_raw, samples, self.config, fold_index, "target", cancer_tgt, manifest
        )

        val_ids = get_source_ids_for_fold(manifest, fold_index, "source_fold_val")
        test_ids = get_source_ids_for_fold(manifest, fold_index, "source_test")
        tgt_all_ids = tables.x_target.index.astype(str).tolist()

        src_val_m = metrics_for_ids(tables.y_source, src_pred_raw, val_ids)
        src_test_m = metrics_for_ids(tables.y_source, src_pred_raw, test_ids)
        tgt_m = metrics_for_ids(tables.y_target, tgt_pred_raw, tgt_all_ids)

        dist_m = compute_distribution_metrics(source_latent, target_latent)

        fold_dir = layout.ensure_fold_dir(fold_index)
        plot_tsne_domain(
            source_latent,
            target_latent,
            self.config.random_seed,
            fold_dir / "tsne_domain_mixing.png",
        )
        kmeans_m = None
        if self.registry.is_available:
            combined_ids = sorted(set(source_latent) | set(target_latent))
            if policy == "exclude":
                combined_ids = self.registry.samples_with_cancer_type(combined_ids)
            labels = [cancer_all[sid] for sid in combined_ids]
            plot_tsne_cancer_type(
                source_latent,
                target_latent,
                labels,
                combined_ids,
                self.config.random_seed,
                fold_dir / "tsne_cancer_type.png",
            )
            combined = {**source_latent, **target_latent}
            kmeans_m = compute_kmeans_cancer_type_metrics(
                combined, cancer_all, self.config.random_seed
            )

        src_meta_rows = []
        for sid in tables.x_source.index.astype(str):
            src_meta_rows.append(
                {
                    "sample_id": sid,
                    "domain": "source",
                    "split": source_split_for_fold(manifest, fold_index, sid),
                    "response_label": int(tables.y_source.loc[sid, "response"]),
                    "cancer_type": cancer_type_label(sid, cancer_src, policy),
                    "fold": fold_index,
                    "seed": self.config.random_seed,
                    "drug": self.config.drug,
                }
            )
        tgt_meta_rows = []
        for sid in tables.x_target.index.astype(str):
            tgt_meta_rows.append(
                {
                    "sample_id": sid,
                    "domain": "target",
                    "target_role": manifest.target_assignments[sid],
                    "response_label": int(tables.y_target.loc[sid, "response"]),
                    "cancer_type": cancer_type_label(sid, cancer_tgt, policy),
                    "fold": fold_index,
                    "seed": self.config.random_seed,
                    "drug": self.config.drug,
                }
            )
        import pandas as pd

        self.writer.write_fold_outputs(
            fold_index=fold_index,
            source_latent=source_latent,
            target_latent=target_latent,
            source_latent_meta=pd.DataFrame(src_meta_rows),
            target_latent_meta=pd.DataFrame(tgt_meta_rows),
            source_pred=src_pred,
            target_pred=tgt_pred,
            source_val_metrics=src_val_m,
            source_test_metrics=src_test_m,
            target_metrics=tgt_m,
            dist_metrics=dist_m,
            kmeans_metrics=kmeans_m,
            tsne_domain_path=fold_dir / "tsne_domain_mixing.png",
            tsne_cancer_path=(
                fold_dir / "tsne_cancer_type.png" if self.registry.is_available else None
            ),
        )
