"""MACD crossover strategy -- classic trend-following approach.

BUY when MACD line crosses above the signal line (bullish crossover).
SELL when MACD line crosses below the signal line (bearish crossover).

One of the most widely used technical strategies; included here as a
solid algorithmic baseline alongside momentum and mean reversion.
"""

from __future__ import annotations

import pandas as pd

from trading_agent.models.signals import Signal, SignalType
from trading_agent.strategies.base import BaseStrategy
from trading_agent.utils.indicators import compute_macd


class MACDCrossoverStrategy(BaseStrategy):
    """MACD line / signal line crossover strategy.

    Parameters
    ----------
    fast:
        Fast EMA period.
    slow:
        Slow EMA period.
    signal:
        Signal line EMA period.
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def evaluate(self, data: pd.DataFrame) -> Signal:
        if data.empty or "close" not in data.columns:
            return Signal(signal_type=SignalType.HOLD, confidence=0.0, strategy_name="macd")

        if len(data) < self.slow + self.signal_period:
            return Signal(signal_type=SignalType.HOLD, confidence=0.0, strategy_name="macd")

        macd_line, signal_line, histogram = compute_macd(
            data["close"], fast=self.fast, slow=self.slow, signal=self.signal_period
        )

        # Check for crossover in the last two bars
        if len(histogram) < 2:
            return Signal(signal_type=SignalType.HOLD, confidence=0.0, strategy_name="macd")

        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]

        meta = {
            "macd": float(macd_line.iloc[-1]),
            "signal_line": float(signal_line.iloc[-1]),
            "histogram": float(current_hist),
        }

        # Bullish crossover: histogram goes from negative to positive
        if prev_hist < 0 and current_hist >= 0:
            confidence = min(abs(current_hist) / (abs(prev_hist) + abs(current_hist) + 1e-9), 1.0)
            return Signal(
                signal_type=SignalType.BUY,
                confidence=confidence,
                strategy_name="macd",
                metadata=meta,
            )

        # Bearish crossover: histogram goes from positive to negative
        if prev_hist > 0 and current_hist <= 0:
            confidence = min(abs(current_hist) / (abs(prev_hist) + abs(current_hist) + 1e-9), 1.0)
            return Signal(
                signal_type=SignalType.SELL,
                confidence=confidence,
                strategy_name="macd",
                metadata=meta,
            )

        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            strategy_name="macd",
            metadata=meta,
        )
