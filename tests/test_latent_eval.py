"""Tests for latent_eval metrics."""

from __future__ import annotations

import numpy as np

from ssda_latent.latent_eval import (
    calculate_fid,
    compute_distribution_metrics,
    tsne_perplexity,
)


def test_distribution_metrics_finite() -> None:
    rng = np.random.default_rng(0)
    src = {f"s{i}": rng.normal(size=8).tolist() for i in range(20)}
    tgt = {f"t{i}": rng.normal(size=8).tolist() for i in range(15)}
    m = compute_distribution_metrics(src, tgt)
    assert m["source_n"] == 20
    assert m["target_n"] == 15
    assert np.isfinite(m["fid_source_target"])
    assert np.isfinite(m["mmd_source_target"])
    assert np.isfinite(m["wasserstein_source_target"])


def test_tsne_perplexity_small_n() -> None:
    assert tsne_perplexity(10) == 3.0
    assert tsne_perplexity(100) == 30.0
    assert tsne_perplexity(2) == 2.0


def test_fid_same_distribution_low() -> None:
    x = np.random.default_rng(1).normal(size=(50, 4))
    fid = calculate_fid(x, x)
    assert fid < 1.0
