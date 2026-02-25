"""Mean-reversion strategy using Bollinger Bands."""

from __future__ import annotations

import pandas as pd

from trading_agent.models.signals import Signal, SignalType
from trading_agent.strategies.base import BaseStrategy
from trading_agent.utils.indicators import compute_bollinger_bands


class MeanReversionStrategy(BaseStrategy):
    """Buy when price touches the lower Bollinger Band, sell at the upper band.

    Parameters
    ----------
    window:
        Rolling window for the moving average.
    num_std:
        Number of standard deviations for the bands.
    """

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self.window = window
        self.num_std = num_std

    def evaluate(self, data: pd.DataFrame) -> Signal:
        if data.empty or len(data) < self.window:
            return Signal(signal_type=SignalType.HOLD, confidence=0.0)

        upper, mid, lower = compute_bollinger_bands(
            data["close"], window=self.window, num_std=self.num_std
        )
        latest_close = data["close"].iloc[-1]
        latest_upper = upper.iloc[-1]
        latest_lower = lower.iloc[-1]
        band_width = latest_upper - latest_lower

        if band_width == 0:
            return Signal(signal_type=SignalType.HOLD, confidence=0.0)

        if latest_close <= latest_lower:
            confidence = (latest_lower - latest_close) / band_width
            return Signal(
                signal_type=SignalType.BUY,
                confidence=min(confidence, 1.0),
                metadata={"bb_lower": latest_lower, "close": latest_close},
            )

        if latest_close >= latest_upper:
            confidence = (latest_close - latest_upper) / band_width
            return Signal(
                signal_type=SignalType.SELL,
                confidence=min(confidence, 1.0),
                metadata={"bb_upper": latest_upper, "close": latest_close},
            )

        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            metadata={"bb_mid": mid.iloc[-1], "close": latest_close},
        )
