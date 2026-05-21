"""Source/target data splits for SSDA latent extension."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ssda_latent.config import ExperimentConfig
from ssda_latent.data_loading import ExpressionTables

SOURCE_SPLIT_TRAIN = "source_fold_train"
SOURCE_SPLIT_VAL = "source_fold_val"
SOURCE_SPLIT_TEST = "source_test"
SOURCE_TRAIN_VAL_POOL = "source_train_val"


@dataclass(frozen=True)
class FoldIndices:
    fold_index: int
    train_ids: frozenset[str]
    val_ids: frozenset[str]


@dataclass(frozen=True)
class SampleSplit:
    sample_id: str
    domain: Literal["source", "target"]
    response_label: int
    source_split: str | None
    target_role: str | None
    fold_index: int | None


@dataclass(frozen=True)
class SplitManifest:
    source_test_ids: frozenset[str]
    folds: tuple[FoldIndices, ...]
    target_assignments: dict[str, str]
    source_assignments: dict[str, str]

    def all_samples(self, tables: ExpressionTables) -> list[SampleSplit]:
        rows: list[SampleSplit] = []
        for sid in tables.x_source.index.astype(str):
            split = self.source_assignments[sid]
            fold_idx = None
            for fold in self.folds:
                if sid in fold.train_ids:
                    fold_idx = fold.fold_index
                    break
                if sid in fold.val_ids:
                    fold_idx = fold.fold_index
                    break
            rows.append(
                SampleSplit(
                    sample_id=sid,
                    domain="source",
                    response_label=int(tables.y_source.loc[sid, "response"]),
                    source_split=split,
                    target_role=None,
                    fold_index=fold_idx,
                )
            )
        for sid in tables.x_target.index.astype(str):
            rows.append(
                SampleSplit(
                    sample_id=sid,
                    domain="target",
                    response_label=int(tables.y_target.loc[sid, "response"]),
                    source_split=None,
                    target_role=self.target_assignments[sid],
                    fold_index=None,
                )
            )
        return rows


def validate_n_shot(y: pd.DataFrame, n_shot: int, label: str) -> None:
    for cls in (0, 1):
        count = int((y["response"] == cls).sum())
        if count < n_shot:
            raise ValueError(f"{label}: class {cls} has only {count} samples, need n_shot={n_shot}")


def assign_target_roles(
    y_target: pd.DataFrame,
    n_shot: int,
    random_seed: int,
) -> dict[str, str]:
    """Mirror experiment_shot.py target few-shot + test definition."""
    validate_n_shot(y_target, n_shot, "target_full")
    y_train, y_val = train_test_split(
        y_target,
        test_size=0.2,
        random_state=random_seed,
    )

    validate_n_shot(y_train, n_shot, "target_train")
    validate_n_shot(y_val, n_shot, "target_val")

    random.seed(random_seed)
    s0_train = random.sample(y_train[y_train["response"] == 0].index.astype(str).tolist(), n_shot)
    s1_train = random.sample(y_train[y_train["response"] == 1].index.astype(str).tolist(), n_shot)
    labeled_train = set(s0_train + s1_train)

    random.seed(random_seed)
    s0_val = random.sample(y_val[y_val["response"] == 0].index.astype(str).tolist(), n_shot)
    s1_val = random.sample(y_val[y_val["response"] == 1].index.astype(str).tolist(), n_shot)
    labeled_val = set(s0_val + s1_val)

    assignments: dict[str, str] = {}
    for sid in y_train.index.astype(str):
        assignments[sid] = (
            "target_labeled_train" if sid in labeled_train else "target_unlabeled_train"
        )
    for sid in y_val.index.astype(str):
        assignments[sid] = "target_labeled_val" if sid in labeled_val else "target_unlabeled_val"

    return assignments


def target_test_ids(assignments: dict[str, str]) -> frozenset[str]:
    """Mirror experiment_shot.py: test = all target minus train labeled only."""
    labeled_train = {sid for sid, role in assignments.items() if role == "target_labeled_train"}
    return frozenset(sid for sid in assignments if sid not in labeled_train)


def build_split_manifest(tables: ExpressionTables, config: ExperimentConfig) -> SplitManifest:
    y_source = tables.y_source
    source_ids = tables.x_source.index.astype(str).tolist()
    labels = y_source.loc[source_ids, "response"].values

    train_val_ids, test_ids = train_test_split(
        source_ids,
        test_size=config.source_test_size,
        random_state=config.random_seed,
        stratify=labels,
    )
    source_test_ids = frozenset(str(i) for i in test_ids)
    train_val_ids = [str(i) for i in train_val_ids]
    y_train_val = y_source.loc[train_val_ids]

    skf = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_seed,
    )
    folds: list[FoldIndices] = []
    source_assignments: dict[str, str] = {sid: SOURCE_SPLIT_TEST for sid in source_test_ids}
    for sid in train_val_ids:
        source_assignments[sid] = SOURCE_TRAIN_VAL_POOL

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_val_ids, y_train_val["response"])):
        train_ids = frozenset(train_val_ids[i] for i in tr_idx)
        val_ids = frozenset(train_val_ids[i] for i in va_idx)
        folds.append(FoldIndices(fold_index=fold_idx, train_ids=train_ids, val_ids=val_ids))

    target_assignments = assign_target_roles(
        tables.y_target,
        config.n_shot,
        config.random_seed,
    )

    _validate_manifest(source_assignments, target_assignments, folds, source_test_ids, config)
    return SplitManifest(
        source_test_ids=source_test_ids,
        folds=tuple(folds),
        target_assignments=target_assignments,
        source_assignments=source_assignments,
    )


def _validate_manifest(
    source_assignments: dict[str, str],
    target_assignments: dict[str, str],
    folds: list[FoldIndices],
    source_test_ids: frozenset[str],
    config: ExperimentConfig,
) -> None:
    for fold in folds:
        assert fold.train_ids.isdisjoint(fold.val_ids)
        assert source_test_ids.isdisjoint(fold.train_ids)
        assert source_test_ids.isdisjoint(fold.val_ids)
    lt = [k for k, v in target_assignments.items() if v == "target_labeled_train"]
    assert len(lt) == config.n_shot * 2


def source_split_for_fold(manifest: SplitManifest, fold: int, sample_id: str) -> str:
    if sample_id in manifest.source_test_ids:
        return SOURCE_SPLIT_TEST
    fold_info = manifest.folds[fold]
    if sample_id in fold_info.train_ids:
        return SOURCE_SPLIT_TRAIN
    if sample_id in fold_info.val_ids:
        return SOURCE_SPLIT_VAL
    return SOURCE_TRAIN_VAL_POOL


def manifest_to_source_split_df(manifest: SplitManifest, tables: ExpressionTables) -> pd.DataFrame:
    rows = []
    for sid in tables.x_source.index.astype(str):
        split = manifest.source_assignments.get(sid, SOURCE_SPLIT_TEST)
        rows.append(
            {
                "sample_id": sid,
                "response_label": int(tables.y_source.loc[sid, "response"]),
                "split": split,
            }
        )
    return pd.DataFrame(rows)


def manifest_to_target_split_df(manifest: SplitManifest, tables: ExpressionTables) -> pd.DataFrame:
    test_ids = target_test_ids(manifest.target_assignments)
    rows = []
    for sid in tables.x_target.index.astype(str):
        rows.append(
            {
                "sample_id": sid,
                "response_label": int(tables.y_target.loc[sid, "response"]),
                "target_role": manifest.target_assignments[sid],
                "in_target_test": sid in test_ids,
            }
        )
    return pd.DataFrame(rows)


def get_source_ids_for_fold(manifest: SplitManifest, fold: int, split_name: str) -> list[str]:
    if split_name == SOURCE_SPLIT_TEST:
        return sorted(manifest.source_test_ids)
    fold_info = manifest.folds[fold]
    if split_name == SOURCE_SPLIT_TRAIN:
        return sorted(fold_info.train_ids)
    if split_name == SOURCE_SPLIT_VAL:
        return sorted(fold_info.val_ids)
    raise ValueError(split_name)


def get_target_ids_by_role(manifest: SplitManifest, role: str) -> list[str]:
    return sorted(sid for sid, r in manifest.target_assignments.items() if r == role)
