"""Low-level file I/O utilities."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(obj: dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_pickle(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as f:
        pickle.dump(obj, f)


def read_pickle(path: str) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def ensure_clean_dir(path: str, overwrite: bool = True) -> None:
    p = Path(path)
    if p.exists():
        if not overwrite:
            raise FileExistsError(f"output_dir exists and overwrite=False: {path}")
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    p.mkdir(parents=True, exist_ok=True)
