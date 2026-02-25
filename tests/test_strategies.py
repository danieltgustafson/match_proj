"""Tests for trading strategies."""

from __future__ import annotations

import pandas as pd

from trading_agent.models.signals import SignalType
from trading_agent.strategies.mean_reversion import MeanReversionStrategy
from trading_agent.strategies.momentum import MomentumStrategy


def test_momentum_hold_on_empty_data() -> None:
    strategy = MomentumStrategy()
    signal = strategy.evaluate(pd.DataFrame())
    assert signal.signal_type is SignalType.HOLD


def test_momentum_hold_on_short_data() -> None:
    strategy = MomentumStrategy(rsi_period=14)
    df = pd.DataFrame({"close": [100.0] * 5})
    signal = strategy.evaluate(df)
    assert signal.signal_type is SignalType.HOLD


def test_momentum_returns_signal(sample_ohlcv: pd.DataFrame) -> None:
    strategy = MomentumStrategy(rsi_period=14)
    signal = strategy.evaluate(sample_ohlcv)
    assert signal.signal_type in (SignalType.BUY, SignalType.SELL, SignalType.HOLD)


def test_mean_reversion_hold_on_empty_data() -> None:
    strategy = MeanReversionStrategy()
    signal = strategy.evaluate(pd.DataFrame())
    assert signal.signal_type is SignalType.HOLD


def test_mean_reversion_returns_signal(sample_ohlcv: pd.DataFrame) -> None:
    strategy = MeanReversionStrategy(window=20, num_std=2.0)
    signal = strategy.evaluate(sample_ohlcv)
    assert signal.signal_type in (SignalType.BUY, SignalType.SELL, SignalType.HOLD)
