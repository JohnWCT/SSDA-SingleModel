"""Centralized artifact writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from codeae_multilabel.config import config_to_dict
from codeae_multilabel.contracts import (
    CodeAEMultilabelConfig,
    DrugIndex,
    PredictionBundle,
    TrainingResult,
)
from codeae_multilabel.data.drug_index import save_drug_list
from codeae_multilabel.io import ensure_clean_dir, write_csv, write_json, write_pickle


class ArtifactWriter:
    def __init__(self, output_dir: str, overwrite: bool = True) -> None:
        self.output_dir = Path(output_dir)
        ensure_clean_dir(str(self.output_dir), overwrite=overwrite)
        self._manifest: list[str] = []

    def _track(self, rel: str) -> None:
        self._manifest.append(rel)

    def write_run_manifest(self) -> None:
        write_json({"artifacts": self._manifest}, str(self.output_dir / "run_manifest.json"))

    def write_config(self, config: CodeAEMultilabelConfig) -> None:
        path = self.output_dir / "config.json"
        write_json(config_to_dict(config), str(path))
        self._track("config.json")

    def write_drug_list(self, drug_index: DrugIndex) -> None:
        path = self.output_dir / "drug_list.csv"
        save_drug_list(drug_index, str(path))
        self._track("drug_list.csv")

    def write_feature_alignment(self, df: pd.DataFrame) -> None:
        path = self.output_dir / "feature_alignment_report.csv"
        write_csv(df, str(path))
        self._track("feature_alignment_report.csv")

    def write_data_alignment(self, df: pd.DataFrame) -> None:
        path = self.output_dir / "data_alignment_report.csv"
        write_csv(df, str(path))
        self._track("data_alignment_report.csv")

    def write_drug_availability(self, df: pd.DataFrame) -> None:
        path = self.output_dir / "drug_availability_report.csv"
        write_csv(df, str(path))
        self._track("drug_availability_report.csv")

    def fold_dir(self, fold_id: int) -> Path:
        d = self.output_dir / f"fold_{fold_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_fold_predictions(self, fold_id: int, bundle: PredictionBundle) -> None:
        fd = self.fold_dir(fold_id)
        write_csv(bundle.source_predictions, str(fd / "source_prediction_results.csv"))
        write_csv(bundle.target_predictions, str(fd / "target_prediction_results.csv"))
        write_csv(bundle.source_metrics_per_drug, str(fd / "source_metrics_per_drug.csv"))
        write_csv(bundle.target_metrics_per_drug, str(fd / "target_metrics_per_drug.csv"))
        write_csv(bundle.source_metrics_summary, str(fd / "source_metrics_summary.csv"))
        write_csv(bundle.target_metrics_summary, str(fd / "target_metrics_summary.csv"))
        for name in (
            "source_prediction_results.csv",
            "target_prediction_results.csv",
            "source_metrics_per_drug.csv",
            "target_metrics_per_drug.csv",
            "source_metrics_summary.csv",
            "target_metrics_summary.csv",
        ):
            self._track(f"fold_{fold_id}/{name}")

    def write_fold_training_result(self, fold_id: int, result: TrainingResult) -> None:
        fd = self.fold_dir(fold_id)
        write_csv(result.train_log, str(fd / "train_log.csv"))
        report = {
            "best_epoch": result.best_epoch,
            "best_metric_name": result.best_metric_name,
            "best_metric_value": result.best_metric_value,
            "best_model_path": result.best_model_path,
        }
        write_json(report, str(fd / "selection_report.json"))
        self._track(f"fold_{fold_id}/train_log.csv")
        self._track(f"fold_{fold_id}/selection_report.json")

    def write_checkpoint_load_report(self, fold_id: int, report: dict[str, Any]) -> None:
        fd = self.fold_dir(fold_id)
        write_json(report, str(fd / "checkpoint_load_report.json"))
        self._track(f"fold_{fold_id}/checkpoint_load_report.json")

    def write_latent(self, fold_id: int, domain: str, latent_df: pd.DataFrame) -> None:
        fd = self.fold_dir(fold_id)
        csv_path = fd / f"{domain}_latent_representation.csv"
        pkl_path = fd / f"{domain}_latent_representation.pkl"
        write_csv(latent_df, str(csv_path))
        obj = {
            row["sample_id"]: row.filter(like="latent_").astype(float).tolist()
            for _, row in latent_df.iterrows()
        }
        write_pickle(obj, str(pkl_path))
        self._track(f"fold_{fold_id}/{domain}_latent_representation.csv")
        self._track(f"fold_{fold_id}/{domain}_latent_representation.pkl")

    def write_combined_latent_pkl(
        self, fold_id: int, source_dict: dict[str, list[float]], target_dict: dict[str, list[float]]
    ) -> None:
        """Full source ∪ target latent vectors (all omics samples)."""
        fd = self.fold_dir(fold_id)
        combined = {**source_dict, **target_dict}
        path = fd / "latent_representation.pkl"
        write_pickle(combined, str(path))
        self._track(f"fold_{fold_id}/latent_representation.pkl")

    def write_pretrain_checkpoint(self, state_dict: dict[str, Any]) -> None:
        import torch

        pre_dir = self.output_dir / "pretrain"
        pre_dir.mkdir(parents=True, exist_ok=True)
        path = pre_dir / "checkpoint.pt"
        torch.save({"shared_encoder": state_dict}, str(path))
        self._track("pretrain/checkpoint.pt")

    def write_fold_summary(self, df: pd.DataFrame) -> None:
        write_csv(df, str(self.output_dir / "fold_summary.csv"))
        self._track("fold_summary.csv")

    def write_summary(self, name: str, df: pd.DataFrame) -> None:
        write_csv(df, str(self.output_dir / name))
        self._track(name)

    def write_fold_csv(self, fold_id: int, name: str, df: pd.DataFrame) -> None:
        fd = self.fold_dir(fold_id)
        write_csv(df, str(fd / name))
        self._track(f"fold_{fold_id}/{name}")
