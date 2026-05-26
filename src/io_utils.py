from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def write_parquet_partitioned(df: pd.DataFrame, dir_path: Path, name: str = "data") -> Path:
    ensure_dir(dir_path)
    out = dir_path / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return out


def read_parquet_dir(dir_path: Path, sample_n: int | None = None) -> pd.DataFrame:
    if dir_path.is_file():
        files = [dir_path]
    else:
        files = sorted(dir_path.glob("*.parquet"))
        if not files:
            nested = sorted(dir_path.glob("**/*.parquet"))
            files = nested
    if not files:
        raise FileNotFoundError(dir_path)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)
    return df.reset_index(drop=True)
