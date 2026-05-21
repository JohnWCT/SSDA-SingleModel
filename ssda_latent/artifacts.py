"""Write experiment artifacts to disk."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from ssda_latent.config import ExperimentConfig
from ssda_latent.paths import RunLayout


class ArtifactWriter:
    def __init__(self, layout: RunLayout, config: ExperimentConfig) -> None:
        self.layout = layout
        self.config = config
        self.layout.ensure_run_dir()

    def write_config(self, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "drug": self.config.drug,
            "random_seed": self.config.random_seed,
            "n_splits": self.config.n_splits,
            "source_test_size": self.config.source_test_size,
            "encoder": self.config.encoder,
            "n_shot": self.config.n_shot,
        }
        if extra:
            payload.update(extra)
        path = self.layout.run_dir / "config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def write_split_tables(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> None:
        source_df.to_csv(self.layout.run_dir / "source_split.csv", index=False)
        target_df.to_csv(self.layout.run_dir / "target_fewshot_split.csv", index=False)

    def write_cancer_summary(self, df: pd.DataFrame) -> None:
        df.to_csv(self.layout.run_dir / "cancer_type_mapping_summary.csv", index=False)

    def save_latent_pkl(self, latent: dict[str, list[float]], path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(latent, f)

    def write_fold_outputs(
        self,
        fold_index: int,
        source_latent: dict[str, list[float]],
        target_latent: dict[str, list[float]],
        source_latent_meta: pd.DataFrame,
        target_latent_meta: pd.DataFrame,
        source_pred: pd.DataFrame,
        target_pred: pd.DataFrame,
        source_val_metrics: dict[str, float],
        source_test_metrics: dict[str, float],
        target_metrics: dict[str, float],
        dist_metrics: dict[str, float],
        kmeans_metrics: dict[str, float] | None,
        tsne_domain_path: Path | None,
        tsne_cancer_path: Path | None,
    ) -> None:
        fold_dir = self.layout.ensure_fold_dir(fold_index)
        self.save_latent_pkl(source_latent, fold_dir / "source_latent_representation.pkl")
        self.save_latent_pkl(target_latent, fold_dir / "target_latent_representation.pkl")
        source_latent_meta.to_csv(fold_dir / "source_latent_metadata.csv", index=False)
        target_latent_meta.to_csv(fold_dir / "target_latent_metadata.csv", index=False)
        source_pred.to_csv(fold_dir / "source_prediction_results.csv", index=False)
        target_pred.to_csv(fold_dir / "target_prediction_results.csv", index=False)
        pd.DataFrame([source_val_metrics]).to_csv(fold_dir / "source_val_metrics.csv", index=False)
        pd.DataFrame([source_test_metrics]).to_csv(
            fold_dir / "source_test_metrics.csv", index=False
        )
        pd.DataFrame([target_metrics]).to_csv(
            fold_dir / "target_prediction_metrics.csv", index=False
        )
        row = {
            **dist_metrics,
            "fold": fold_index,
            "seed": self.config.random_seed,
            "drug": self.config.drug,
        }
        pd.DataFrame([row]).to_csv(fold_dir / "latent_distribution_metrics.csv", index=False)
        if kmeans_metrics is not None:
            km = {
                **kmeans_metrics,
                "fold": fold_index,
                "seed": self.config.random_seed,
                "drug": self.config.drug,
            }
            pd.DataFrame([km]).to_csv(fold_dir / "kmeans_cancer_type_metrics.csv", index=False)
        if tsne_domain_path is not None:
            pass
        if tsne_cancer_path is not None:
            pass

    def write_summaries(
        self,
        metrics_summary: pd.DataFrame,
        latent_summary: pd.DataFrame,
        kmeans_summary: pd.DataFrame | None,
    ) -> None:
        metrics_summary.to_csv(self.layout.run_dir / "metrics_summary.csv", index=False)
        latent_summary.to_csv(self.layout.run_dir / "latent_metrics_summary.csv", index=False)
        if kmeans_summary is not None:
            kmeans_summary.to_csv(
                self.layout.run_dir / "kmeans_cancer_type_summary.csv", index=False
            )
