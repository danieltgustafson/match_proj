"""Data cleaning and feature engineering utilities."""

from __future__ import annotations

import pandas as pd


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Basic OHLCV data cleaning.

    - Drops rows with any NaN in OHLCV columns.
    - Sorts by index (date).
    - Ensures numeric types.
    """
    ohlcv = ["open", "high", "low", "close", "volume"]
    cols = [c for c in ohlcv if c in df.columns]
    df = df.dropna(subset=cols)
    df = df.sort_index()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=cols)


def add_returns(df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
    """Add a `returns` column with simple percentage returns."""
    df = df.copy()
    df["returns"] = df[column].pct_change()
    return df


def add_log_returns(df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
    """Add a `log_returns` column."""
    import numpy as np

    df = df.copy()
    df["log_returns"] = np.log(df[column] / df[column].shift(1))
    return df
