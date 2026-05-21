"""Load processed source/target expression tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ssda_latent.config import ExperimentConfig


@dataclass(frozen=True)
class ExpressionTables:
    x_source: pd.DataFrame
    y_source: pd.DataFrame
    x_target: pd.DataFrame
    y_target: pd.DataFrame


def load_tables(config: ExperimentConfig) -> ExpressionTables:
    base = Path(config.data_path) / config.drug
    x_source = pd.read_csv(
        base / "source_data" / f"source_scaled{config.gene_suffix}.csv",
        index_col=0,
    ).T
    y_source = pd.read_csv(base / "source_data" / "source_meta_data.csv", index_col=0)
    x_target = pd.read_csv(
        base / "target_data" / f"target_scaled{config.gene_suffix}.csv",
        index_col=0,
    ).T
    y_target = pd.read_csv(base / "target_data" / "target_meta_data.csv", index_col=0)
    y_source.index = y_source.index.astype(str)
    y_target.index = y_target.index.astype(str)
    x_source.index = x_source.index.astype(str)
    x_target.index = x_target.index.astype(str)
    return ExpressionTables(
        x_source=x_source,
        y_source=y_source,
        x_target=x_target,
        y_target=y_target,
    )
