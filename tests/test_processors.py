"""Tests for data processors."""

from __future__ import annotations

import pandas as pd

from trading_agent.data.processors import add_log_returns, add_returns, clean_ohlcv


def test_clean_ohlcv_drops_nan(sample_ohlcv: pd.DataFrame) -> None:
    dirty = sample_ohlcv.copy()
    dirty.loc[dirty.index[0], "close"] = float("nan")
    cleaned = clean_ohlcv(dirty)
    assert cleaned["close"].isna().sum() == 0
    assert len(cleaned) == len(dirty) - 1


def test_add_returns(sample_ohlcv: pd.DataFrame) -> None:
    result = add_returns(sample_ohlcv)
    assert "returns" in result.columns
    assert result["returns"].iloc[0] != result["returns"].iloc[0]  # NaN check


def test_add_log_returns(sample_ohlcv: pd.DataFrame) -> None:
    result = add_log_returns(sample_ohlcv)
    assert "log_returns" in result.columns
