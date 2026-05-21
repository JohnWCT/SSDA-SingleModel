#!/usr/bin/env python3
"""
Optional: reproducible Python equivalent of Datasets/n-shot.ipynb.

Writes labeled/unlabeled train & val splits and a held-out test folder.
Not read by experiment_shot.py; useful for baselines / fixed n-shot CSVs / inspection.

Preserves notebook typo: subdirectory name "labled_target_data".
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="N-shot split (n-shot.ipynb).")
    parser.add_argument("--drug", type=str, required=True)
    parser.add_argument("--n", type=int, default=3, help="Shots per class (train and val).")
    parser.add_argument("--random-state", type=int, default=5)
    parser.add_argument("--gene-set-tag", type=str, default="tp4k")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("./Datasets/processedData"),
    )
    args = parser.parse_args()
    random.seed(args.random_state)

    processed = args.processed_dir.resolve()
    drug = args.drug
    n = args.n
    tag = args.gene_set_tag

    tp4k_path = processed / drug / "target_data" / tag
    all_train_X = pd.read_csv(tp4k_path / "X_train_target.csv", index_col=0)
    all_train_Y = pd.read_csv(tp4k_path / "Y_train_target.csv", index_col=0)
    all_val_X = pd.read_csv(tp4k_path / "X_val_target.csv", index_col=0)
    all_val_Y = pd.read_csv(tp4k_path / "Y_val_target.csv", index_col=0)

    idx0 = all_train_Y[all_train_Y["response"] == 0].index.tolist()
    idx1 = all_train_Y[all_train_Y["response"] == 1].index.tolist()
    if len(idx0) < n or len(idx1) < n:
        raise ValueError(
            f"Not enough training samples per class for n={n}: class0={len(idx0)}, class1={len(idx1)}"
        )
    sample_0 = random.sample(idx0, n)
    sample_1 = random.sample(idx1, n)

    labeled_train_Y = all_train_Y.loc[sample_0 + sample_1]
    labeled_train_X = all_train_X.loc[:, labeled_train_Y.index.tolist()]
    unlabeled_train_Y = all_train_Y.drop(sample_0 + sample_1, axis=0)
    unlabeled_train_X = all_train_X.loc[:, unlabeled_train_Y.index.tolist()]

    idx0v = all_val_Y[all_val_Y["response"] == 0].index.tolist()
    idx1v = all_val_Y[all_val_Y["response"] == 1].index.tolist()
    if len(idx0v) < n or len(idx1v) < n:
        raise ValueError(
            f"Not enough val samples per class for n={n}: class0={len(idx0v)}, class1={len(idx1v)}"
        )
    sample_0_val = random.sample(idx0v, n)
    sample_1_val = random.sample(idx1v, n)

    labeled_val_Y = all_val_Y.loc[sample_0_val + sample_1_val]
    labeled_val_X = all_val_X.loc[:, labeled_val_Y.index.tolist()]
    unlabeled_val_Y = all_val_Y.drop(sample_0_val + sample_1_val, axis=0)
    unlabeled_val_X = all_val_X.loc[:, unlabeled_val_Y.index.tolist()]

    save_path = processed / drug / f"{n}-shot" / tag
    labeled_path = save_path / "labled_target_data"
    unlabeled_path = save_path / "unlabeled_target_data"
    labeled_path.mkdir(parents=True, exist_ok=True)
    unlabeled_path.mkdir(parents=True, exist_ok=True)

    labeled_train_Y.to_csv(labeled_path / "Y_train_target.csv")
    labeled_train_X.to_csv(labeled_path / "X_train_target.csv")
    unlabeled_train_Y.to_csv(unlabeled_path / "Y_train_target.csv")
    unlabeled_train_X.to_csv(unlabeled_path / "X_train_target.csv")

    labeled_val_Y.to_csv(labeled_path / "Y_val_target.csv")
    labeled_val_X.to_csv(labeled_path / "X_val_target.csv")
    unlabeled_val_X.to_csv(unlabeled_path / "X_val_target.csv")
    unlabeled_val_Y.to_csv(unlabeled_path / "Y_val_target.csv")

    test_path = processed / drug / "target_data"
    all_Y = pd.read_csv(test_path / "target_meta_data.csv", index_col=0)
    all_X = pd.read_csv(test_path / f"target_scaled_{tag}.csv", index_col=0)
    test_Y = all_Y.drop(labeled_train_Y.index, axis=0)
    test_X = all_X.loc[:, test_Y.index.tolist()]
    test_X = test_X[test_Y.index]

    save_test_path = save_path / "test_data"
    save_test_path.mkdir(parents=True, exist_ok=True)
    test_X.to_csv(save_test_path / "unlabeled_test_X.csv")
    test_Y.to_csv(save_test_path / "unlabeled_test_Y.csv")

    print(f"Done: n-shot -> {save_path}")


if __name__ == "__main__":
    main()
