"""Yahoo Finance data provider."""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_agent.data.providers.base import BaseProvider


class YahooFinanceProvider(BaseProvider):
    """Fetch historical OHLCV data from Yahoo Finance via the `yfinance` library."""

    def get_historical(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for YahooFinanceProvider. "
                "Install it with: pip install yfinance"
            ) from exc

        ticker = yf.Ticker(symbol)
        df: pd.DataFrame = ticker.history(start=start, end=end, **kwargs)

        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Keep only OHLCV columns that exist
        ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[ohlcv_cols]
