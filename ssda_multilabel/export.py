"""Artifact paths and file writing for multi-label SSDA outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ssda_multilabel.config import MultiLabelConfig
from ssda_multilabel.io import ensure_dir, write_csv, write_pickle
from ssda_multilabel.schemas import DrugIndex, ResponseMatrix, TargetMasks


class ArtifactWriter:
    def __init__(self, root_dir: str, seed: int) -> None:
        self.root = Path(root_dir) / f"seed_{seed}"
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
        source_resp: ResponseMatrix,
        target_resp: ResponseMatrix,
        target_masks: TargetMasks,
        alignment_report: pd.DataFrame,
        nshot_summary: pd.DataFrame,
        cancer_summary: pd.DataFrame,
    ) -> None:
        from ssda_multilabel.drug_index import save_drug_list

        save_drug_list(drug_index, str(self.root / "drug_list.csv"))
        write_csv(alignment_report, self.root / "data_alignment_report.csv")
        write_csv(nshot_summary, self.root / "target_nshot_summary.csv")
        write_csv(cancer_summary, self.root / "cancer_type_mapping_summary.csv")
        self._write_matrix(source_resp, "source")
        self._write_matrix(target_resp, "target")
        self._write_mask(target_masks.observed_mask, "target_observed_mask.csv")
        self._write_mask(target_masks.labeled_mask, "target_labeled_mask.csv")
        self._write_mask(target_masks.unlabeled_mask, "target_unlabeled_mask.csv")

    def _write_matrix(self, rm: ResponseMatrix, prefix: str) -> None:
        df_y = pd.DataFrame(rm.y, index=rm.sample_ids, columns=list(rm.drug_index.drug_ids))
        df_m = pd.DataFrame(rm.mask, index=rm.sample_ids, columns=list(rm.drug_index.drug_ids))
        y_path = self.root / f"{prefix}_response_matrix.csv"
        m_path = self.root / f"{prefix}_response_mask.csv"
        write_csv(df_y.reset_index().rename(columns={"index": "sample_id"}), y_path)
        write_csv(df_m.reset_index().rename(columns={"index": "sample_id"}), m_path)

    def _write_mask(self, mask: np.ndarray[Any, Any], name: str) -> None:
        pd.DataFrame(mask).to_csv(self.root / name, index=False)

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
