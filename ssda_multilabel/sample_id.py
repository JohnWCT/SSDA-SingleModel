"""Sample ID detection and normalization for cross-table joins."""

from __future__ import annotations

import re

# Fixed sample / drug columns (no CLI).
SOURCE_OMICS_SAMPLE_ID_COL = "Sample_ID"
SOURCE_RESPONSE_SAMPLE_ID_COL = "Sample_ID"
TARGET_OMICS_SAMPLE_ID_COL = "tissue_id"
TARGET_RESPONSE_SAMPLE_ID_COL = "Patient_id"
TARGET_RESPONSE_LABEL_COL = "Label"
DRUG_NAME_COL = "drug_name"

_TCGA_PREFIX = re.compile(r"^TCGA-", re.IGNORECASE)


def tcga_segment_count(sample_id: str) -> int:
    return len(str(sample_id).strip().split("-"))


def is_tcga_barcode(sample_id: str) -> bool:
    """Full TCGA barcode has more than 4 hyphen segments."""
    sid = str(sample_id).strip()
    return bool(_TCGA_PREFIX.match(sid)) and tcga_segment_count(sid) > 4


def is_tcga_tissue_key(sample_id: str) -> bool:
    """TCGA tissue id: TCGA-XX-XXXX-XXA (4 segments)."""
    sid = str(sample_id).strip()
    return bool(_TCGA_PREFIX.match(sid)) and tcga_segment_count(sid) == 4


def is_tcga_patient_key(sample_id: str) -> bool:
    """Short TCGA patient key: TCGA-XX-XXXX (3 segments)."""
    sid = str(sample_id).strip()
    return bool(_TCGA_PREFIX.match(sid)) and tcga_segment_count(sid) == 3


def tcga_tissue_key(sample_id: str) -> str:
    """Map TCGA barcode to 4-segment tissue id."""
    sid = str(sample_id).strip()
    parts = sid.split("-")
    if _TCGA_PREFIX.match(sid) and len(parts) >= 4:
        return "-".join(parts[:4])
    return sid


def tcga_patient_key(sample_id: str) -> str:
    """Map TCGA tissue id / barcode to 3-segment patient key."""
    sid = str(sample_id).strip()
    parts = sid.split("-")
    if _TCGA_PREFIX.match(sid) and len(parts) >= 3:
        return "-".join(parts[:3])
    return sid


def sample_match_key(sample_id: str, *, column_hint: str | None = None) -> str:
    """Normalize sample ids so target tissue_id joins target Patient_id.

    - ``tissue_id`` (4-segment) / long barcode -> 3-segment patient key
    - ``Patient_id`` -> same 3-segment key
    - DepMap ``ACH-*`` and other ids -> unchanged
    """
    sid = str(sample_id).strip()
    hint = (column_hint or "").lower()
    if _TCGA_PREFIX.match(sid):
        return tcga_patient_key(sid)
    if hint in {"tissue_id", "barcode_id", "patient_id", "sample_id"} and tcga_segment_count(sid) >= 3:
        return tcga_patient_key(sid)
    return sid


def describe_sample_id_column(values: list[str], column_name: str) -> str:
    """Return a short description for alignment reports."""
    tcga = [v for v in values[:500] if _TCGA_PREFIX.match(str(v))]
    if not tcga:
        return f"{column_name}: non-TCGA ids"
    n_barcode = sum(1 for v in tcga if is_tcga_barcode(v))
    n_tissue = sum(1 for v in tcga if is_tcga_tissue_key(v))
    n_key = sum(1 for v in tcga if is_tcga_patient_key(v))
    return (
        f"{column_name}: TCGA n={len(tcga)} "
        f"(barcode_style={n_barcode}, tissue_style={n_tissue}, patient_key_style={n_key})"
    )
