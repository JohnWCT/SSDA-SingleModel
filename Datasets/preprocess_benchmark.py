#!/usr/bin/env python3
"""
Reproducible Python equivalent of Datasets/benchmark.ipynb.

Produces the four CSV files required by experiment_shot.py:
  processedData/<drug>/source_data/source_scaled_tp4k.csv
  processedData/<drug>/source_data/source_meta_data.csv
  processedData/<drug>/target_data/target_scaled_tp4k.csv
  processedData/<drug>/target_data/target_meta_data.csv
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def _ensure_raw_dir(drug: str, raw_dir: Path) -> Path:
    """Return path raw_dir/<drug>/; unzip raw_dir/<drug>.zip if folder is missing."""
    drug_dir = raw_dir / drug
    if drug_dir.is_dir():
        return drug_dir
    zpath = raw_dir / f"{drug}.zip"
    if zpath.is_file():
        raw_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(raw_dir)
        if not drug_dir.is_dir():
            print(
                f"Warning: after extracting {zpath}, expected directory {drug_dir} not found.",
                file=sys.stderr,
            )
        return drug_dir
    raise FileNotFoundError(
        f"Neither {drug_dir} nor {zpath} exists. Place raw TSVs under rawData/{drug}/ or provide {zpath}."
    )


def run_benchmark(
    drug: str,
    raw_dir: Path,
    processed_dir: Path,
    tsv_suffix: str,
    expression_set_slug: str,
) -> None:
    """
    tsv_suffix: substring in raw filenames, e.g. '_tp4k' -> Source_exprs_resp_z.<drug>_tp4k.tsv
    expression_set_slug: used in output filenames (e.g. tp4k -> source_exp_data_tp4k.csv).
    """
    raw_path = _ensure_raw_dir(drug, raw_dir)
    # Notebook: 'Source_exprs_resp_z.' + drug + '_tp4k.tsv'
    piece = tsv_suffix if str(tsv_suffix).startswith("_") else f"_{tsv_suffix}"
    source_tsv = raw_path / f"Source_exprs_resp_z.{drug}{piece}.tsv"
    target_tsv = raw_path / f"Target_expr_resp_z.{drug}{piece}.tsv"
    if not source_tsv.is_file() or not target_tsv.is_file():
        raise FileNotFoundError(f"Missing TSV: {source_tsv} or {target_tsv}")

    source_save = processed_dir / drug / "source_data"
    target_save = processed_dir / drug / "target_data"
    source_save.mkdir(parents=True, exist_ok=True)
    target_save.mkdir(parents=True, exist_ok=True)

    source_data = pd.read_csv(source_tsv, sep="\t", index_col=0)
    source_meta_data = source_data[["response", "logIC50"]]
    source_data = source_data.drop(["response", "logIC50"], axis=1)
    source_data = source_data.T
    source_data.to_csv(source_save / f"source_exp_data_{expression_set_slug}.csv")

    target_data = pd.read_csv(target_tsv, sep="\t", index_col=0)
    target_meta_data = target_data[["response"]]
    target_data = target_data.drop(["response"], axis=1)
    target_data = target_data.T
    target_data.to_csv(target_save / f"target_exp_data_{expression_set_slug}.csv")

    source_meta_data.to_csv(source_save / "source_meta_data.csv")
    target_meta_data.to_csv(target_save / "target_meta_data.csv")

    # Align with notebook: transpose back to cells x genes before concat + scaler
    source_data = source_data.T
    target_data = target_data.T
    combine = pd.concat([source_data, target_data], axis=0)
    scaler = StandardScaler()
    scaler.fit(combine)

    source_scaled = pd.DataFrame(
        scaler.transform(source_data), columns=source_data.columns, index=source_data.index
    )
    target_scaled = pd.DataFrame(
        scaler.transform(target_data), columns=target_data.columns, index=target_data.index
    )
    source_scaled = source_scaled.T
    target_scaled = target_scaled.T

    source_scaled.to_csv(source_save / f"source_scaled_{expression_set_slug}.csv")
    target_scaled.to_csv(target_save / f"target_scaled_{expression_set_slug}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark preprocessing (benchmark.ipynb).")
    parser.add_argument("--drug", type=str, required=True, help="Drug folder name under rawData, e.g. Gefitinib")
    parser.add_argument(
        "--tsv-suffix",
        type=str,
        default="_tp4k",
        help="Raw filename segment after drug, e.g. _tp4k for ..._Gefitinib_tp4k.tsv",
    )
    parser.add_argument(
        "--gene-set-tag",
        type=str,
        default="tp4k",
        help="Expression-set label used in output filenames (not a single gene symbol).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("./Datasets/rawData"),
        help="Directory containing <drug>/ or <drug>.zip",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("./Datasets/processedData"),
        help="Output root for processedData/<drug>/...",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    run_benchmark(
        drug=args.drug,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        tsv_suffix=args.tsv_suffix,
        expression_set_slug=args.gene_set_tag,
    )
    print(f"Done: benchmark preprocessing for {args.drug} -> {processed_dir / args.drug}")


if __name__ == "__main__":
    main()
