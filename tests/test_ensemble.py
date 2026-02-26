"""Tests for the ensemble strategy."""

from __future__ import annotations

import pandas as pd
import pytest

from trading_agent.models.signals import Signal, SignalType
from trading_agent.strategies.base import BaseStrategy
from trading_agent.strategies.ensemble import EnsembleStrategy


class AlwaysBuyStrategy(BaseStrategy):
    def evaluate(self, data: pd.DataFrame) -> Signal:
        return Signal(signal_type=SignalType.BUY, confidence=0.8, strategy_name="always_buy")


class AlwaysSellStrategy(BaseStrategy):
    def evaluate(self, data: pd.DataFrame) -> Signal:
        return Signal(signal_type=SignalType.SELL, confidence=0.7, strategy_name="always_sell")


class AlwaysHoldStrategy(BaseStrategy):
    def evaluate(self, data: pd.DataFrame) -> Signal:
        return Signal(signal_type=SignalType.HOLD, confidence=0.0, strategy_name="always_hold")


def test_empty_strategies_raises() -> None:
    with pytest.raises(ValueError):
        EnsembleStrategy(strategies={})


def test_unanimous_buy() -> None:
    ensemble = EnsembleStrategy(
        strategies={"a": AlwaysBuyStrategy(), "b": AlwaysBuyStrategy()},
    )
    signal = ensemble.evaluate(pd.DataFrame({"close": [100.0]}))
    assert signal.signal_type is SignalType.BUY
    assert signal.confidence > 0


def test_unanimous_sell() -> None:
    ensemble = EnsembleStrategy(
        strategies={"a": AlwaysSellStrategy(), "b": AlwaysSellStrategy()},
    )
    signal = ensemble.evaluate(pd.DataFrame({"close": [100.0]}))
    assert signal.signal_type is SignalType.SELL


def test_mixed_signals_majority_buy() -> None:
    ensemble = EnsembleStrategy(
        strategies={
            "buy1": AlwaysBuyStrategy(),
            "buy2": AlwaysBuyStrategy(),
            "sell": AlwaysSellStrategy(),
        },
        min_agreement=0.5,
    )
    signal = ensemble.evaluate(pd.DataFrame({"close": [100.0]}))
    assert signal.signal_type is SignalType.BUY


def test_no_consensus_returns_hold() -> None:
    ensemble = EnsembleStrategy(
        strategies={
            "buy": AlwaysBuyStrategy(),
            "sell": AlwaysSellStrategy(),
        },
        min_agreement=0.6,  # need 60% agreement
    )
    signal = ensemble.evaluate(pd.DataFrame({"close": [100.0]}))
    assert signal.signal_type is SignalType.HOLD


def test_strategy_name_is_ensemble() -> None:
    ensemble = EnsembleStrategy(strategies={"a": AlwaysBuyStrategy()})
    signal = ensemble.evaluate(pd.DataFrame({"close": [100.0]}))
    assert signal.strategy_name == "ensemble"


def test_metadata_contains_sub_signals() -> None:
    ensemble = EnsembleStrategy(
        strategies={"a": AlwaysBuyStrategy(), "b": AlwaysSellStrategy()},
    )
    signal = ensemble.evaluate(pd.DataFrame({"close": [100.0]}))
    assert "sub_signals" in signal.metadata
    assert "a" in signal.metadata["sub_signals"]
    assert "b" in signal.metadata["sub_signals"]


def test_evaluate_with_precomputed_signals() -> None:
    ensemble = EnsembleStrategy(strategies={"a": AlwaysBuyStrategy()})
    signals = {
        "a": Signal(signal_type=SignalType.BUY, confidence=0.9),
        "b": Signal(signal_type=SignalType.BUY, confidence=0.7),
    }
    result = ensemble.evaluate_with_signals(signals)
    assert result.signal_type is SignalType.BUY
