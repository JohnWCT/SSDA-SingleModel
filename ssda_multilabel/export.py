"""Artifact paths and file writing for multi-label SSDA outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ssda_multilabel.config import MultiLabelConfig
from ssda_multilabel.io import ensure_dir, write_csv, write_pickle
from ssda_multilabel.schemas import DrugIndex


class ArtifactWriter:
    def __init__(self, root_dir: str, seed: int) -> None:
        del seed  # recorded in config.json and per-row exports; no seed_* folder
        self.root = Path(root_dir)
        ensure_dir(self.root)

    def fold_dir(self, fold_id: int) -> Path:
        p = self.root / f"fold_{fold_id}"
        ensure_dir(p)
        return p

    def write_config(self, config: MultiLabelConfig) -> None:
        config.save_json(self.root / "config.json")

    def write_preparation_artifacts(
        self,
        drug_index: DrugIndex,
        alignment_report: pd.DataFrame,
    ) -> None:
        from ssda_multilabel.drug_index import save_drug_list

        save_drug_list(drug_index, str(self.root / "drug_list.csv"))
        write_csv(alignment_report, self.root / "data_alignment_report.csv")

    def save_model(self, fold_id: int, model: torch.nn.Module) -> Path:
        path = self.fold_dir(fold_id) / "model_final.pth"
        torch.save(model.state_dict(), path)
        return path

    def write_fold_csv(self, fold_id: int, name: str, df: pd.DataFrame) -> None:
        write_csv(df, self.fold_dir(fold_id) / name)

    def write_fold_pkl(self, fold_id: int, name: str, obj: object) -> None:
        write_pickle(obj, self.fold_dir(fold_id) / name)

    def write_summary(self, name: str, df: pd.DataFrame) -> None:
        write_csv(df, self.root / name)
