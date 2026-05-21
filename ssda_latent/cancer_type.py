"""Cancer type metadata alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from ssda_latent.config import ExperimentConfig
from ssda_latent.data_loading import ExpressionTables

UNKNOWN_LABEL = "Unknown"


class SampleIdNormalizer(Protocol):
    def normalize(self, sample_id: str, domain: str) -> str: ...


class DefaultNormalizer:
    def normalize(self, sample_id: str, domain: str) -> str:
        return str(sample_id).strip()


@dataclass(frozen=True)
class CancerTypeAlignmentReport:
    domain: str
    total_expression_samples: int
    matched_samples: int
    missing_in_metadata: int
    extra_in_metadata: int
    unknown_samples: int
    excluded_samples: int


@dataclass(frozen=True)
class CancerTypeRegistry:
    source_map: dict[str, str]
    target_map: dict[str, str]
    reports: tuple[CancerTypeAlignmentReport, ...]
    is_available: bool

    def get(self, sample_id: str, domain: str) -> str:
        sid = str(sample_id)
        if domain == "source":
            return self.source_map.get(sid, UNKNOWN_LABEL)
        return self.target_map.get(sid, UNKNOWN_LABEL)

    def combined_map(self) -> dict[str, str]:
        return {**self.source_map, **self.target_map}

    def samples_with_cancer_type(self, sample_ids: list[str]) -> list[str]:
        """Sample IDs that have an aligned cancer type (exclude policy drops others)."""
        cmap = self.combined_map()
        return [sid for sid in sample_ids if sid in cmap]


def _resolve_col(df: pd.DataFrame, col: str) -> str:
    if col in df.columns:
        return col
    lookup = {str(c).lower(): c for c in df.columns}
    key = col.lower()
    if key in lookup:
        return lookup[key]
    if key == "sample_id" and "sample_id" in lookup:
        return lookup["sample_id"]
    if key == "cancer_type" and "cancer_type" in lookup:
        return lookup["cancer_type"]
    raise ValueError(f"column {col} not found in {list(df.columns)}")


def load_cancer_type_table(
    path: str,
    sample_id_col: str,
    cancer_type_col: str,
    domain: str,
    normalizer: SampleIdNormalizer,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    sid_col = _resolve_col(df, sample_id_col)
    ct_col = _resolve_col(df, cancer_type_col)
    out = df[[sid_col, ct_col]].dropna()
    out[sid_col] = out[sid_col].astype(str).map(lambda x: normalizer.normalize(x, domain))
    out[ct_col] = out[ct_col].astype(str)
    out = out.drop_duplicates(subset=[sid_col], keep="first")
    return out.rename(columns={sid_col: "sample_id", ct_col: "cancer_type"})


def align_domain(
    expression_ids: list[str],
    cancer_df: pd.DataFrame | None,
    domain: str,
    policy: str,
) -> tuple[dict[str, str], CancerTypeAlignmentReport]:
    total = len(expression_ids)
    if cancer_df is None:
        default_map = {sid: UNKNOWN_LABEL for sid in expression_ids}
        return default_map, CancerTypeAlignmentReport(
            domain=domain,
            total_expression_samples=total,
            matched_samples=0,
            missing_in_metadata=total,
            extra_in_metadata=0,
            unknown_samples=total,
            excluded_samples=0,
        )
    ct_map = dict(zip(cancer_df["sample_id"], cancer_df["cancer_type"]))
    expr_set = set(expression_ids)
    meta_set = set(ct_map.keys())
    matched = expr_set & meta_set
    missing = expr_set - meta_set
    extra = meta_set - expr_set
    mapping: dict[str, str] = {}
    unknown_count = 0
    excluded = 0
    for sid in expression_ids:
        if sid in ct_map:
            mapping[sid] = ct_map[sid]
        elif policy == "unknown":
            mapping[sid] = UNKNOWN_LABEL
            unknown_count += 1
        else:
            excluded += 1
    return mapping, CancerTypeAlignmentReport(
        domain=domain,
        total_expression_samples=total,
        matched_samples=len(matched),
        missing_in_metadata=len(missing),
        extra_in_metadata=len(extra),
        unknown_samples=unknown_count,
        excluded_samples=excluded,
    )


def build_registry(
    tables: ExpressionTables,
    config: ExperimentConfig,
    normalizer: SampleIdNormalizer | None = None,
) -> CancerTypeRegistry:
    norm = normalizer or DefaultNormalizer()
    src_ids = tables.x_source.index.astype(str).tolist()
    tgt_ids = tables.x_target.index.astype(str).tolist()
    src_df = (
        load_cancer_type_table(
            config.source_cancer_type_path,
            config.sample_id_col,
            config.cancer_type_col,
            "source",
            norm,
        )
        if config.source_cancer_type_path
        else None
    )
    tgt_df = (
        load_cancer_type_table(
            config.target_cancer_type_path,
            config.sample_id_col,
            config.cancer_type_col,
            "target",
            norm,
        )
        if config.target_cancer_type_path
        else None
    )
    src_map, src_rep = align_domain(src_ids, src_df, "source", config.missing_cancer_type_policy)
    tgt_map, tgt_rep = align_domain(tgt_ids, tgt_df, "target", config.missing_cancer_type_policy)
    is_available = (
        config.source_cancer_type_path is not None or config.target_cancer_type_path is not None
    )
    return CancerTypeRegistry(
        source_map=src_map,
        target_map=tgt_map,
        reports=(src_rep, tgt_rep),
        is_available=is_available,
    )


def cancer_type_label(
    sample_id: str, mapping: dict[str, str], policy: str
) -> str | None:
    sid = str(sample_id)
    if sid in mapping:
        return mapping[sid]
    if policy == "unknown":
        return UNKNOWN_LABEL
    return None


def reports_to_dataframe(registry: CancerTypeRegistry) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in registry.reports])
