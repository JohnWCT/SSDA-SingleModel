#!/usr/bin/env python3
"""
Optional: reproducible Python equivalent of Datasets/split_data.ipynb.

Outputs train/val CSVs under processedData/<drug>/source_data/<gene-set-tag>/
and processedData/<drug>/target_data/<gene-set-tag>/.

Not read by experiment_shot.py; useful for baselines / fixed splits / inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/val split (split_data.ipynb).")
    parser.add_argument("--drug", type=str, required=True)
    parser.add_argument("--gene-set-tag", type=str, default="tp4k", help="Subfolder name, e.g. tp4k")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=5)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("./Datasets/processedData"),
    )
    args = parser.parse_args()
    processed = args.processed_dir.resolve()
    tag = args.gene_set_tag

    source_path = processed / args.drug / "source_data"
    target_path = processed / args.drug / "target_data"

    source_X = pd.read_csv(source_path / f"source_scaled_{tag}.csv", index_col=0)
    source_Y = pd.read_csv(source_path / "source_meta_data.csv", index_col=0)
    target_X = pd.read_csv(target_path / f"target_scaled_{tag}.csv", index_col=0)
    target_Y = pd.read_csv(target_path / "target_meta_data.csv", index_col=0)
    # Notebook: transpose so rows = cells (samples), cols = genes — matches meta rows.
    source_X = source_X.T
    target_X = target_X.T

    source_X_train, source_X_val, source_Y_train, source_Y_val = train_test_split(
        source_X, source_Y, test_size=args.test_size, random_state=args.random_state
    )
    target_X_train, target_X_val, target_Y_train, target_Y_val = train_test_split(
        target_X, target_Y, test_size=args.test_size, random_state=args.random_state
    )

    source_save = source_path / tag
    target_save = target_path / tag
    source_save.mkdir(parents=True, exist_ok=True)
    target_save.mkdir(parents=True, exist_ok=True)

    source_X_train = source_X_train.T
    source_X_val = source_X_val.T
    target_X_train = target_X_train.T
    target_X_val = target_X_val.T

    source_X_train.to_csv(source_save / "X_train_source.csv")
    source_X_val.to_csv(source_save / "X_val_source.csv")
    source_Y_train.to_csv(source_save / "Y_train_source.csv")
    source_Y_val.to_csv(source_save / "Y_val_source.csv")

    target_X_train.to_csv(target_save / "X_train_target.csv")
    target_X_val.to_csv(target_save / "X_val_target.csv")
    target_Y_train.to_csv(target_save / "Y_train_target.csv")
    target_Y_val.to_csv(target_save / "Y_val_target.csv")

    print(f"Done: split_data -> {source_save}, {target_save}")


if __name__ == "__main__":
    main()
