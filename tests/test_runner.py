"""Tests for the trading runner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trading_agent.execution.paper_trading import PaperBroker
from trading_agent.models.signals import Signal, SignalType
from trading_agent.runner import TradingRunner
from trading_agent.strategies.base import BaseStrategy


class StubStrategy(BaseStrategy):
    """Always returns a BUY signal."""

    def evaluate(self, data: pd.DataFrame) -> Signal:
        return Signal(
            signal_type=SignalType.BUY,
            confidence=0.7,
            strategy_name="stub",
        )


class StubProvider:
    """Returns synthetic OHLCV data."""

    def get_historical(self, symbol: str, **kwargs) -> pd.DataFrame:
        import numpy as np

        n = 60
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        return pd.DataFrame({
            "open": close + 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [50000] * n,
        })


def test_run_once_paper() -> None:
    broker = PaperBroker(initial_cash=100_000)
    runner = TradingRunner(
        broker=broker,
        data_provider=StubProvider(),
        strategy=StubStrategy(),
        watchlist=["AAPL", "MSFT"],
    )
    result = runner.run_once()
    assert result.symbols_evaluated == 2
    assert result.signals_generated >= 0
    assert len(result.errors) == 0


def test_run_once_with_empty_data() -> None:
    class EmptyProvider:
        def get_historical(self, symbol, **kwargs):
            return pd.DataFrame()

    broker = PaperBroker()
    runner = TradingRunner(
        broker=broker,
        data_provider=EmptyProvider(),
        strategy=StubStrategy(),
        watchlist=["AAPL"],
    )
    result = runner.run_once()
    assert result.symbols_evaluated == 1
    assert result.orders_submitted == 0


def test_run_scheduled_limited() -> None:
    broker = PaperBroker()
    runner = TradingRunner(
        broker=broker,
        data_provider=StubProvider(),
        strategy=StubStrategy(),
        watchlist=["AAPL"],
    )
    results = runner.run_scheduled(interval_seconds=0, max_iterations=2)
    assert len(results) == 2


def test_from_settings_paper_mode() -> None:
    from trading_agent.config import Settings

    settings = Settings(paper_trading=True, alpha_vantage_api_key="test-key")
    runner = TradingRunner.from_settings(
        settings=settings,
        watchlist=["AAPL"],
    )
    assert isinstance(runner.broker, PaperBroker)
