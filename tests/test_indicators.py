"""Tests for technical indicators."""

from __future__ import annotations

import pandas as pd

from trading_agent.utils.indicators import (
    compute_bollinger_bands,
    compute_ema,
    compute_macd,
    compute_rsi,
    compute_sma,
)


def test_compute_rsi_range(sample_ohlcv: pd.DataFrame) -> None:
    rsi = compute_rsi(sample_ohlcv["close"], period=14)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_compute_sma_length(sample_ohlcv: pd.DataFrame) -> None:
    sma = compute_sma(sample_ohlcv["close"], window=20)
    assert len(sma) == len(sample_ohlcv)
    assert sma.iloc[:19].isna().all()
    assert sma.iloc[19:].notna().all()


def test_compute_ema_length(sample_ohlcv: pd.DataFrame) -> None:
    ema = compute_ema(sample_ohlcv["close"], span=20)
    assert len(ema) == len(sample_ohlcv)


def test_bollinger_bands_order(sample_ohlcv: pd.DataFrame) -> None:
    upper, mid, lower = compute_bollinger_bands(sample_ohlcv["close"])
    valid_idx = upper.dropna().index
    assert (upper[valid_idx] >= mid[valid_idx]).all()
    assert (mid[valid_idx] >= lower[valid_idx]).all()


def test_compute_macd_shapes(sample_ohlcv: pd.DataFrame) -> None:
    macd_line, signal_line, histogram = compute_macd(sample_ohlcv["close"])
    assert len(macd_line) == len(sample_ohlcv)
    assert len(signal_line) == len(sample_ohlcv)
    assert len(histogram) == len(sample_ohlcv)
