"""Main experiment runner."""

from __future__ import annotations

from datetime import datetime, timezone

from ssda_latent.artifacts import ArtifactWriter
from ssda_latent.cancer_type import build_registry, reports_to_dataframe
from ssda_latent.config import ExperimentConfig
from ssda_latent.data_loading import load_tables
from ssda_latent.dataloader_factory import build_fold_dataloaders
from ssda_latent.export_pipeline import ExportPipeline
from ssda_latent.paths import RunLayout
from ssda_latent.seed import SeedManager
from ssda_latent.split import (
    build_split_manifest,
    manifest_to_source_split_df,
    manifest_to_target_split_df,
)
from ssda_latent.summary import (
    aggregate_fold_metrics,
    aggregate_kmeans_metrics,
    aggregate_latent_metrics,
)
from ssda_latent.training_adapter import train_fold


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> None:
        SeedManager.set_all(self.config.random_seed)
        layout = RunLayout.from_config(self.config)
        writer = ArtifactWriter(layout, self.config)
        tables = load_tables(self.config)
        manifest = build_split_manifest(tables, self.config)
        registry = build_registry(tables, self.config)

        writer.write_config(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latent_dim": self.config.latent_dim,
                "source_n": len(tables.x_source),
                "target_n": len(tables.x_target),
            }
        )
        writer.write_split_tables(
            manifest_to_source_split_df(manifest, tables),
            manifest_to_target_split_df(manifest, tables),
        )
        if registry.is_available:
            writer.write_cancer_summary(reports_to_dataframe(registry))

        exporter = ExportPipeline(self.config, registry, writer)

        for fold in range(self.config.n_splits):
            loaders = build_fold_dataloaders(tables, manifest, fold, self.config)
            fold_dir = layout.ensure_fold_dir(fold)
            training = train_fold(self.config, loaders, fold, fold_dir)
            exporter.run(fold, training, tables, manifest, layout)

        writer.write_summaries(
            aggregate_fold_metrics(layout.run_dir, self.config.n_splits),
            aggregate_latent_metrics(layout.run_dir, self.config.n_splits),
            aggregate_kmeans_metrics(layout.run_dir, self.config.n_splits),
        )
