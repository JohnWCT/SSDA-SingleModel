#!/usr/bin/env python3
"""Patch DAPL CSV columns: PRISM, GDSC, CCLE Sample_ID; TCGA tissue_id."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ssda_multilabel.drug_normalize import load_broad_id_to_name
from ssda_multilabel.sample_id import TARGET_OMICS_SAMPLE_ID_COL, tcga_tissue_key


def patch_prism(prism_path: Path, drug_smiles_path: Path) -> None:
    df = pd.read_csv(prism_path)
    if "depmap_id" in df.columns and "Sample_ID" not in df.columns:
        df = df.rename(columns={"depmap_id": "Sample_ID"})
    mapping = load_broad_id_to_name(str(drug_smiles_path))
    if "broad_id" not in df.columns:
        raise ValueError(f"{prism_path} missing broad_id column")
    df["drug_name"] = df["broad_id"].astype(str).map(mapping)
    missing = int(df["drug_name"].isna().sum())
    if missing:
        raise ValueError(f"{missing} PRISM rows could not map broad_id -> drug_name")
    df["drug_name"] = df["drug_name"].astype(str).str.strip().str.lower()
    df.to_csv(prism_path, index=False)
    print(f"Patched {prism_path}: Sample_ID + drug_name ({len(df)} rows)")


def patch_gdsc_sample_id(path: Path) -> None:
    """GDSC response: ModelID (DepMap) -> Sample_ID to match pretrain_ccle."""
    df = pd.read_csv(path)
    if "Sample_ID" in df.columns:
        print(f"Skip {path}: already has Sample_ID")
        return
    if "ModelID" not in df.columns:
        raise ValueError(f"{path} missing ModelID column")
    df = df.rename(columns={"ModelID": "Sample_ID"})
    df.to_csv(path, index=False)
    print(f"Patched {path}: ModelID -> Sample_ID ({len(df)} rows)")


def patch_pretrain_sample_id(path: Path) -> None:
    """CCLE pretrain omics: first column -> Sample_ID."""
    df = pd.read_csv(path)
    first = str(df.columns[0])
    if first == "Sample_ID":
        print(f"Skip {path}: already has Sample_ID")
        return
    df = df.rename(columns={first: "Sample_ID"})
    df.to_csv(path, index=False)
    print(f"Patched {path}: renamed {first!r} -> Sample_ID ({len(df)} rows)")


def patch_pretrain_tissue_id(path: Path) -> None:
    """TCGA pretrain omics: first column -> tissue_id."""
    df = pd.read_csv(path)
    first = str(df.columns[0])
    if first == TARGET_OMICS_SAMPLE_ID_COL:
        print(f"Skip {path}: already has {TARGET_OMICS_SAMPLE_ID_COL}")
        return
    df = df.rename(columns={first: TARGET_OMICS_SAMPLE_ID_COL})
    df.to_csv(path, index=False)
    print(f"Patched {path}: renamed {first!r} -> {TARGET_OMICS_SAMPLE_ID_COL} ({len(df)} rows)")


def patch_tcga_tissue_id(path: Path, *, normalize_values: bool, chunksize: int = 50_000) -> None:
    """Rename first ID column to tissue_id; optionally truncate TCGA barcodes to 4 segments."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    first_chunk = True
    n_rows = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        sid_col = str(chunk.columns[0])
        if sid_col != TARGET_OMICS_SAMPLE_ID_COL:
            chunk = chunk.rename(columns={sid_col: TARGET_OMICS_SAMPLE_ID_COL})
        if normalize_values:
            chunk[TARGET_OMICS_SAMPLE_ID_COL] = (
                chunk[TARGET_OMICS_SAMPLE_ID_COL].astype(str).str.strip().map(tcga_tissue_key)
            )
        chunk.to_csv(tmp, index=False, mode="w" if first_chunk else "a", header=first_chunk)
        first_chunk = False
        n_rows += len(chunk)
    tmp.replace(path)
    print(
        f"Patched {path}: -> {TARGET_OMICS_SAMPLE_ID_COL} "
        f"(normalize={normalize_values}, {n_rows} rows)"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dapl-root",
        type=Path,
        default=Path("/home/wasijk/Drug/DAPL-master"),
    )
    args = p.parse_args()
    root = args.dapl_root
    patch_prism(
        root / "data_Winnie/PRISM_drug_sensitivity.csv",
        root / "data_Winnie/drug_smiles.csv",
    )
    patch_pretrain_sample_id(root / "data/pretrain_ccle.csv")
    patch_gdsc_sample_id(root / "data/GDSC2_fitted_dose_response_MaxScreen_raw.csv")
    legacy_gdsc = (
        root
        / "data/GDSC2_fitted_dose_response_27Oct23 from GDSC MaxScreen threshold ModelID966 drug230 samples201288.csv"
    )
    if legacy_gdsc.is_file():
        patch_gdsc_sample_id(legacy_gdsc)
    patch_pretrain_tissue_id(root / "data/TCGA/pretrain_tcga.csv")
    patch_tcga_tissue_id(
        root / "data_Winnie/TCGA_impact_hotspot.csv",
        normalize_values=True,
    )


if __name__ == "__main__":
    main()
