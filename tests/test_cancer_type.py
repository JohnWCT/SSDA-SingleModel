"""Tests for cancer type alignment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ssda_latent.cancer_type import (
    DefaultNormalizer,
    align_domain,
    cancer_type_label,
    load_cancer_type_table,
)


def test_align_unknown_policy(tmp_path: Path) -> None:
    meta = tmp_path / "ct.csv"
    pd.DataFrame({"Sample_ID": ["s1"], "Cancer_type": ["LUAD"]}).to_csv(meta, index=False)
    df = load_cancer_type_table(
        str(meta), "Sample_ID", "Cancer_type", "source", DefaultNormalizer()
    )
    mapping, report = align_domain(["s1", "s2"], df, "source", "unknown")
    assert mapping["s1"] == "LUAD"
    assert mapping["s2"] == "Unknown"
    assert report.missing_in_metadata == 1


def test_align_exclude_policy() -> None:
    meta = pd.DataFrame({"Sample_ID": ["s1"], "Cancer_type": ["LUAD"]})
    df = meta.rename(columns={"Sample_ID": "sample_id", "Cancer_type": "cancer_type"})
    mapping, report = align_domain(["s1", "s2"], df, "source", "exclude")
    assert mapping == {"s1": "LUAD"}
    assert "s2" not in mapping
    assert report.excluded_samples == 1
    assert cancer_type_label("s2", mapping, "exclude") is None
    assert cancer_type_label("s2", mapping, "unknown") == "Unknown"
