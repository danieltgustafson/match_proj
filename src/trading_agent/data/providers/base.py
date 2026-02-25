"""Abstract base class for market data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseProvider(ABC):
    """Interface that all data providers must implement."""

    @abstractmethod
    def get_historical(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame for *symbol* over the given date range.

        Expected columns: open, high, low, close, volume
        Index: DatetimeIndex
        """
