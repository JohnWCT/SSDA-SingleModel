"""Latent distribution metrics and t-SNE (reuses ssda_latent implementations)."""

from __future__ import annotations

from ssda_latent.latent_eval import (
    compute_distribution_metrics,
    compute_kmeans_cancer_type_metrics,
    plot_tsne_cancer_type,
    plot_tsne_domain,
    tsne_perplexity,
)

__all__ = [
    "compute_distribution_metrics",
    "compute_kmeans_cancer_type_metrics",
    "plot_tsne_cancer_type",
    "plot_tsne_domain",
    "tsne_perplexity",
]
