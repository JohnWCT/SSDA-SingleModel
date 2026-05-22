"""Low-level file I/O without domain logic."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)


def write_pickle(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("wb") as f:
        pickle.dump(obj, f)


def read_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
