"""Base agent class that all trading agents inherit from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trading_agent.config import Settings
from trading_agent.models.signals import Signal
from trading_agent.strategies.base import BaseStrategy


@dataclass
class BacktestResult:
    """Container for backtest output metrics."""

    trades: list[dict[str, Any]] = field(default_factory=list)
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0

    def summary(self) -> str:
        return (
            f"Trades: {len(self.trades)} | "
            f"Return: {self.total_return_pct:.2f}% | "
            f"Sharpe: {self.sharpe_ratio:.2f} | "
            f"Max DD: {self.max_drawdown_pct:.2f}% | "
            f"Win Rate: {self.win_rate:.1f}%"
        )


class BaseAgent(ABC):
    """Abstract base class for trading agents.

    Subclasses must implement `fetch_data` and `execute_signal`.
    The core `run` loop is: fetch data -> generate signal -> execute.
    """

    def __init__(self, settings: Settings, strategy: BaseStrategy) -> None:
        self.settings = settings
        self.strategy = strategy

    @abstractmethod
    def fetch_data(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """Retrieve market data for *symbol*."""

    @abstractmethod
    def execute_signal(self, signal: Signal) -> dict[str, Any]:
        """Send an order based on the given signal."""

    def run(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        """Fetch data, evaluate strategy, and execute the resulting signal."""
        data = self.fetch_data(symbol, **kwargs)
        signal = self.strategy.evaluate(data)
        return self.execute_signal(signal)

    def backtest(
        self,
        symbol: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> BacktestResult:
        """Run a simple vectorised backtest over historical data.

        This is a placeholder implementation -- override for more
        sophisticated backtesting logic.
        """
        data = self.fetch_data(symbol, start=start, end=end, **kwargs)
        _ = self.strategy.evaluate(data)
        return BacktestResult()
