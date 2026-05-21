"""Latent distribution metrics and t-SNE plots."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

UNKNOWN_LABEL = "Unknown"


def tsne_perplexity(n_samples: int) -> float:
    """Perplexity safe for small combined latent sets (sklearn default 30 needs n > 30)."""
    if n_samples < 2:
        return 2.0
    return float(min(30, max(2, (n_samples - 1) // 3)))


def _to_matrix(latent_dict: dict[str, list[float]], sample_ids: list[str]) -> NDArray[np.float64]:
    return np.asarray([latent_dict[sid] for sid in sample_ids], dtype=np.float64)


def calculate_fid(source: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    diff = mu_s - mu_t
    cov_s = np.cov(source, rowvar=False) + np.eye(source.shape[1]) * 1e-6
    cov_t = np.cov(target, rowvar=False) + np.eye(target.shape[1]) * 1e-6
    from scipy.linalg import sqrtm

    covmean = sqrtm(cov_s @ cov_t)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff.dot(diff) + np.trace(cov_s + cov_t - 2 * covmean))
    return fid


def calculate_mmd(
    source: NDArray[np.float64], target: NDArray[np.float64], gamma: float | None = None
) -> float:
    if gamma is None:
        gamma = 1.0 / source.shape[1]
    n = min(500, source.shape[0], target.shape[0])
    rng = np.random.default_rng(0)
    xs = source[rng.choice(source.shape[0], n, replace=False)]
    xt = target[rng.choice(target.shape[0], n, replace=False)]

    def kernel(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        xx = np.sum(x * x, axis=1, keepdims=True)
        yy = np.sum(y * y, axis=1, keepdims=True)
        xy = x @ y.T
        return cast(NDArray[np.float64], np.exp(-gamma * (xx - 2 * xy + yy.T)))

    k_ss = kernel(xs, xs).mean()
    k_tt = kernel(xt, xt).mean()
    k_st = kernel(xs, xt).mean()
    return float(k_ss + k_tt - 2 * k_st)


def calculate_wasserstein(source: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    from scipy.stats import wasserstein_distance

    dists = [wasserstein_distance(source[:, j], target[:, j]) for j in range(source.shape[1])]
    return float(np.mean(dists))


def compute_distribution_metrics(
    source_latent: dict[str, list[float]],
    target_latent: dict[str, list[float]],
) -> dict[str, float]:
    src_ids = sorted(source_latent.keys())
    tgt_ids = sorted(target_latent.keys())
    src = _to_matrix(source_latent, src_ids)
    tgt = _to_matrix(target_latent, tgt_ids)
    return {
        "source_n": float(len(src_ids)),
        "target_n": float(len(tgt_ids)),
        "fid_source_target": calculate_fid(src, tgt),
        "mmd_source_target": calculate_mmd(src, tgt),
        "wasserstein_source_target": calculate_wasserstein(src, tgt),
    }


def _latent_vector(
    source_latent: dict[str, list[float]],
    target_latent: dict[str, list[float]],
    sample_id: str,
) -> list[float]:
    if sample_id in source_latent:
        return source_latent[sample_id]
    return target_latent[sample_id]


def _combined_matrix(
    source_latent: dict[str, list[float]],
    target_latent: dict[str, list[float]],
) -> tuple[NDArray[np.float64], list[str], list[str]]:
    ids = sorted(set(source_latent) | set(target_latent))
    domains = ["source" if sid in source_latent else "target" for sid in ids]
    mat = np.asarray(
        [_latent_vector(source_latent, target_latent, sid) for sid in ids],
        dtype=np.float64,
    )
    return mat, ids, domains


def plot_tsne_domain(
    source_latent: dict[str, list[float]],
    target_latent: dict[str, list[float]],
    random_state: int,
    output_path: Path,
) -> None:
    mat, _, domains = _combined_matrix(source_latent, target_latent)
    emb = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        perplexity=tsne_perplexity(mat.shape[0]),
    ).fit_transform(mat)
    fig, ax = plt.subplots(figsize=(8, 6))
    for domain, color in [("source", "tab:blue"), ("target", "tab:orange")]:
        mask = [d == domain for d in domains]
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6, label=domain, c=color)
    ax.legend()
    ax.set_title("t-SNE domain mixing")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tsne_cancer_type(
    source_latent: dict[str, list[float]],
    target_latent: dict[str, list[float]],
    cancer_labels: list[str],
    sample_ids: list[str],
    random_state: int,
    output_path: Path,
) -> None:
    mat = np.asarray(
        [_latent_vector(source_latent, target_latent, sid) for sid in sample_ids],
        dtype=np.float64,
    )
    emb = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        perplexity=tsne_perplexity(mat.shape[0]),
    ).fit_transform(mat)
    fig, ax = plt.subplots(figsize=(10, 6))
    unique = sorted(set(cancer_labels))
    for ct in unique:
        mask = [lab == ct for lab in cancer_labels]
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6, label=ct)
    ax.legend(markerscale=2, fontsize=6, loc="best")
    ax.set_title("t-SNE cancer type")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_kmeans_cancer_type_metrics(
    combined_latent: dict[str, list[float]],
    cancer_map: dict[str, str],
    random_state: int,
) -> dict[str, float]:
    ids = []
    labels = []
    for sid, ct in cancer_map.items():
        if sid not in combined_latent:
            continue
        if ct == UNKNOWN_LABEL:
            continue
        ids.append(sid)
        labels.append(ct)
    if len(ids) < 2:
        return {
            "k_eff": float("nan"),
            "samples_used": float(len(ids)),
            "ari": float("nan"),
            "nmi": float("nan"),
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
            "n_cancer_types": 0.0,
        }
    x = _to_matrix(combined_latent, ids)
    unique_labels = sorted(set(labels))
    k = len(unique_labels)
    k_eff = int(max(2, min(k, len(ids) - 1)))
    pred = KMeans(n_clusters=k_eff, random_state=random_state, n_init=10).fit_predict(x)
    y_true = np.asarray(labels)
    metrics: dict[str, float] = {
        "k_eff": float(k_eff),
        "samples_used": float(len(ids)),
        "ari": float(adjusted_rand_score(y_true, pred)),
        "nmi": float(normalized_mutual_info_score(y_true, pred)),
        "n_cancer_types": float(k),
        "silhouette": float("nan"),
        "calinski_harabasz": float("nan"),
        "davies_bouldin": float("nan"),
    }
    with contextlib.suppress(Exception):
        metrics["silhouette"] = float(silhouette_score(x, pred))
    with contextlib.suppress(Exception):
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(x, pred))
    with contextlib.suppress(Exception):
        metrics["davies_bouldin"] = float(davies_bouldin_score(x, pred))
    return metrics
