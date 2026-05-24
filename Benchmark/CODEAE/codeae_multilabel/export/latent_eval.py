"""Latent distribution and clustering metrics (SSDA-compatible)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ssda_latent.latent_eval import (  # noqa: E402
    compute_distribution_metrics,
    compute_kmeans_cancer_type_metrics,
    plot_tsne_cancer_type,
    plot_tsne_domain,
)

__all__ = [
    "compute_distribution_metrics",
    "compute_kmeans_cancer_type_metrics",
    "plot_tsne_cancer_type",
    "plot_tsne_domain",
    "build_cancer_map",
]


def build_cancer_map(cancer_type_df: pd.DataFrame | None) -> dict[str, str]:
    if cancer_type_df is None or cancer_type_df.empty:
        return {}
    if "sample_id" not in cancer_type_df.columns or "cancer_type" not in cancer_type_df.columns:
        return {}
    return dict(
        zip(
            cancer_type_df["sample_id"].astype(str),
            cancer_type_df["cancer_type"].astype(str),
        )
    )
