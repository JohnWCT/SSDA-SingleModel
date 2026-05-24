"""Typed data contracts for CODE-AE multilabel pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class CodeAEMultilabelConfig:
    task_type: Literal["classification", "regression"]
    source_omics_path: str
    target_omics_path: str
    source_response_path: Optional[str]
    target_response_path: Optional[str]
    source_sample_col: str
    target_sample_col: str
    target_response_sample_col: str
    drug_col: str
    source_response_col: str
    target_response_col: str
    method: str
    pretrain_checkpoint: Optional[str]
    output_dir: str
    overwrite: bool
    batch_size: int
    epochs: int
    lr: float
    seed: int
    n_splits: int
    source_test_size: float
    metric: Optional[str]
    reg_loss: Literal["mae"]
    prediction_threshold: float
    regression_binary_threshold: float
    source_cancer_type_path: Optional[str]
    target_cancer_type_path: Optional[str]
    cancer_type_col: Optional[str]
    latent_dim: int
    encoder_hidden_dims: tuple[int, ...]
    classifier_hidden_dims: tuple[int, ...]
    pretrain_num_epochs: int
    train_num_epochs: int
    norm_flag: bool
    alpha: float
    dop: float
    decay_coefficient: float
    early_stopping_tolerance: int
    duplicate_strategy: Literal["mean", "median", "first", "error"]
    device: str
    retrain_flag: bool = True
    es_flag: bool = False
    max_samples: Optional[int] = None
    max_drugs: Optional[int] = None
    freeze_encoder_initially: bool = True
    progressive_unfreeze: bool = True
    finetune_domain_loss: Literal["none", "coral", "mmd", "adversarial"] = "none"
    finetune_domain_lambda: float = 0.0
    finetune_domain_warmup_epochs: int = 0
    finetune_wgan_gp: float = 10.0
    finetune_gen_every: int = 5

    @property
    def n_drugs_hint(self) -> Optional[int]:
        return None

    @property
    def uses_finetune_unlabeled_loss(self) -> bool:
        return self.finetune_domain_loss != "none"


@dataclass(frozen=True)
class OmicsTable:
    x: pd.DataFrame
    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    domain: Literal["source", "target"]


@dataclass(frozen=True)
class DrugIndex:
    drug_ids: tuple[str, ...]
    drug_to_index: dict[str, int]
    index_to_drug: dict[int, str]

    @property
    def n_drugs(self) -> int:
        return len(self.drug_ids)


@dataclass(frozen=True)
class ResponseMatrix:
    y: NDArray[np.float32]
    mask: NDArray[np.float32]
    sample_ids: tuple[str, ...]
    drug_index: DrugIndex
    domain: Literal["source", "target"]
    label_semantics: Literal["binary", "continuous"]


@dataclass(frozen=True)
class PreparedPretrainData:
    source_omics: OmicsTable
    target_omics: OmicsTable
    feature_alignment: pd.DataFrame


@dataclass(frozen=True)
class SourceFold:
    fold_id: int
    train_sample_ids: tuple[str, ...]
    val_sample_ids: tuple[str, ...]
    test_sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class PreparedFineTuneData:
    source_omics: OmicsTable
    target_omics: OmicsTable
    source_response: ResponseMatrix
    target_response: ResponseMatrix
    drug_index: DrugIndex
    drug_index_full: DrugIndex
    folds: tuple[SourceFold, ...]
    cancer_type_table: Optional[pd.DataFrame]
    cancer_type_summary: Optional[pd.DataFrame]
    feature_alignment: pd.DataFrame


@dataclass
class TrainingResult:
    fold_id: int
    best_model_path: str
    best_epoch: int
    best_metric_name: str
    best_metric_value: float
    train_log: pd.DataFrame


@dataclass
class PredictionBundle:
    source_predictions: pd.DataFrame
    target_predictions: pd.DataFrame
    source_metrics_per_drug: pd.DataFrame
    target_metrics_per_drug: pd.DataFrame
    source_metrics_summary: pd.DataFrame
    target_metrics_summary: pd.DataFrame
