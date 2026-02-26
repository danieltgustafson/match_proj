"""Momentum-based trading strategy using RSI."""

from __future__ import annotations

import pandas as pd

from trading_agent.models.signals import Signal, SignalType
from trading_agent.strategies.base import BaseStrategy
from trading_agent.utils.indicators import compute_rsi


class MomentumStrategy(BaseStrategy):
    """Generate buy/sell signals based on RSI thresholds.

    Parameters
    ----------
    rsi_period:
        Look-back window for RSI calculation.
    rsi_overbought:
        RSI level above which we consider the asset overbought (sell).
    rsi_oversold:
        RSI level below which we consider the asset oversold (buy).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
    ) -> None:
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def evaluate(self, data: pd.DataFrame) -> Signal:
        if data.empty or len(data) < self.rsi_period + 1:
            return Signal(signal_type=SignalType.HOLD, confidence=0.0, strategy_name="momentum")

        rsi_series = compute_rsi(data["close"], period=self.rsi_period)
        latest_rsi = rsi_series.iloc[-1]

        if latest_rsi <= self.rsi_oversold:
            confidence = (self.rsi_oversold - latest_rsi) / self.rsi_oversold
            return Signal(
                signal_type=SignalType.BUY,
                confidence=min(confidence, 1.0),
                strategy_name="momentum",
                metadata={"rsi": latest_rsi},
            )

        if latest_rsi >= self.rsi_overbought:
            confidence = (latest_rsi - self.rsi_overbought) / (100.0 - self.rsi_overbought)
            return Signal(
                signal_type=SignalType.SELL,
                confidence=min(confidence, 1.0),
                strategy_name="momentum",
                metadata={"rsi": latest_rsi},
            )

        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            strategy_name="momentum",
            metadata={"rsi": latest_rsi},
        )
