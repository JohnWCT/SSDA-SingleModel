"""Training log dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class EpochLog:
    fold_id: int
    epoch: int
    source_prediction_loss: float
    codeae_loss_total: float
    total_loss: float
    val_metric_name: str
    val_metric_value: float
    selected_metric_name: str
    selected_metric_value: float


@dataclass
class FoldTrainState:
    fold_id: int
    best_epoch: int
    best_metric_name: str
    best_metric_value: float
    best_model_path: str
    epoch_logs: list[EpochLog]


def epoch_logs_to_dataframe(logs: list[EpochLog]) -> pd.DataFrame:
    return pd.DataFrame([asdict(x) for x in logs])
