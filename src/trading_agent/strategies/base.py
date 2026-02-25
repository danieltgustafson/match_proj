"""Base strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from trading_agent.models.signals import Signal


class BaseStrategy(ABC):
    """All strategies implement `evaluate` which returns a Signal."""

    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> Signal:
        """Analyse *data* and return a trading signal."""
