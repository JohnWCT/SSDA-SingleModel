"""Cancer type metadata alignment (DAPL sample-info / Winnie tables)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ssda_multilabel.sample_id import tcga_tissue_key

UNKNOWN = "Unknown"

# (sample_id_column, cancer_type_column, optional key normalizer name)
_PROFILE_COLUMNS: dict[str, tuple[str, str, str | None]] = {
    "ccle_sample_info": ("Unnamed: 0", "cancer_type", None),
    "xena_sample_info": ("Unnamed: 0", "cancer_type", None),
    "ccle_cancer_type": ("Sample_ID", "Cancer_type", None),
    "tcga_cancer_type": ("Sample_ID", "Cancer_type", "tissue_id"),
}


def infer_dapl_root(*paths: str, default: str = "/workspace/DAPL-master") -> str:
    for raw in paths:
        if not raw:
            continue
        p = str(raw)
        marker = "DAPL-master"
        if marker in p:
            return p.split(marker)[0] + marker
    return default


def resolve_cancer_type_paths(
    source_omics_path: str,
    target_omics_path: str,
    dapl_root: str | None = None,
) -> tuple[str | None, str | None]:
    """Pick DAPL cancer-type tables from omics paths (pretrain vs Winnie impact)."""
    root = Path(dapl_root or infer_dapl_root(source_omics_path, target_omics_path))
    sop = source_omics_path.lower()
    top = target_omics_path.lower()

    if "pretrain_ccle" in sop:
        src = root / "data/ccle_sample_info_df.csv"
    elif "ccle_impact" in sop or "ccle" in sop:
        src = root / "data_Winnie/CCLE_cancer_type.csv"
    else:
        src = None

    if "pretrain_tcga" in top:
        tgt = root / "data/TCGA/xena_sample_info_df.csv"
    elif "tcga_impact" in top:
        tgt = root / "data_Winnie/TCGA_cancer_type.csv"
    else:
        tgt = None

    return (str(src) if src and src.is_file() else None, str(tgt) if tgt and tgt.is_file() else None)


def _detect_profile(path: str) -> str:
    name = Path(path).name.lower()
    if "ccle_sample_info" in name:
        return "ccle_sample_info"
    if "xena_sample_info" in name:
        return "xena_sample_info"
    if "ccle_cancer_type" in name:
        return "ccle_cancer_type"
    if "tcga_cancer_type" in name:
        return "tcga_cancer_type"
    raise ValueError(f"unsupported cancer type file: {path}")


def _normalize_sample_key(sample_id: str, mode: str | None) -> str:
    sid = str(sample_id).strip()
    if mode == "tissue_id":
        return tcga_tissue_key(sid)
    return sid


def load_cancer_type_mapping(path: str) -> dict[str, str]:
    """Load sample_id -> cancer_type; TCGA Winnie tables keyed by 4-segment tissue_id."""
    profile = _detect_profile(path)
    sid_col, ct_col, key_mode = _PROFILE_COLUMNS[profile]
    df = pd.read_csv(path)
    lookup = {str(c).lower(): c for c in df.columns}
    sid_c = sid_col if sid_col in df.columns else lookup.get(sid_col.lower())
    ct_c = ct_col if ct_col in df.columns else lookup.get(ct_col.lower())
    if ct_c is None and profile == "ccle_sample_info":
        ct_c = lookup.get("primary_disease")
    if ct_c is None and profile == "xena_sample_info":
        ct_c = lookup.get("_primary_disease")
    if sid_c is None or ct_c is None:
        raise ValueError(f"{path}: missing columns for profile {profile}")
    out: dict[str, str] = {}
    for sid, ct in zip(df[sid_c].astype(str), df[ct_c].astype(str)):
        key = _normalize_sample_key(sid, key_mode)
        if not key or (isinstance(ct, float) and pd.isna(ct)):
            continue
        out[key] = str(ct).strip()
    return out


def load_and_align_cancer_types(
    source_ids: list[str],
    target_ids: list[str],
    source_path: str | None,
    target_path: str | None,
    sample_id_col: str,
    cancer_type_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del sample_id_col, cancer_type_col  # columns resolved per DAPL profile in load_cancer_type_mapping
    rows: list[dict[str, str]] = []

    def _align(path: str | None, domain: str, ids: list[str]) -> None:
        if path is None:
            for sid in ids:
                rows.append({"sample_id": sid, "domain": domain, "cancer_type": UNKNOWN})
            return
        mapping = load_cancer_type_mapping(path)
        for sid in ids:
            rows.append(
                {
                    "sample_id": sid,
                    "domain": domain,
                    "cancer_type": mapping.get(str(sid).strip(), UNKNOWN),
                }
            )

    _align(source_path, "source", source_ids)
    _align(target_path, "target", target_ids)
    full = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "domain": d,
                "n_samples": int((full["domain"] == d).sum()),
                "n_unknown": int(((full["domain"] == d) & (full["cancer_type"] == UNKNOWN)).sum()),
                "cancer_type_path": source_path if d == "source" else target_path,
            }
            for d in ("source", "target")
        ]
    )
    return full, summary
