"""Alpha Vantage data provider for stocks and crypto.

Docs: https://www.alphavantage.co/documentation/

Supports:
- TIME_SERIES_DAILY / TIME_SERIES_INTRADAY for equities
- DIGITAL_CURRENCY_DAILY for crypto
"""

from __future__ import annotations

from typing import Any, Optional

import httpx
import pandas as pd

from trading_agent.data.providers.base import BaseProvider

_BASE_URL = "https://www.alphavantage.co/query"

# Maps Alpha Vantage column prefixes to clean names
_OHLCV_RENAME = {
    "1. open": "open",
    "2. high": "high",
    "3. low": "low",
    "4. close": "close",
    "5. volume": "volume",
}


class AlphaVantageError(Exception):
    """Raised when the Alpha Vantage API returns an error or unexpected payload."""


class AlphaVantageProvider(BaseProvider):
    """Fetch OHLCV data from the Alpha Vantage REST API.

    Parameters
    ----------
    api_key:
        Your Alpha Vantage API key.  Loaded from settings if not provided.
    base_url:
        Override for testing / proxying.
    """

    def __init__(self, api_key: str, base_url: str = _BASE_URL) -> None:
        if not api_key:
            raise ValueError(
                "Alpha Vantage API key is required. "
                "Set ALPHA_VANTAGE_API_KEY in your .env file."
            )
        self.api_key = api_key
        self.base_url = base_url

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_historical(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return daily OHLCV for *symbol*.

        For crypto symbols (e.g. ``BTC``, ``ETH``) pass ``asset_type="crypto"``
        in *kwargs* to hit the digital-currency endpoint.
        """
        asset_type = kwargs.pop("asset_type", "stock")
        if asset_type == "crypto":
            market = kwargs.pop("market", "USD")
            return self.get_crypto_daily(symbol, market=market, start=start, end=end)
        return self.get_stock_daily(symbol, start=start, end=end, **kwargs)

    def get_stock_daily(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        outputsize: str = "compact",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch daily stock prices via ``TIME_SERIES_DAILY``."""
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }
        data = self._request(params)
        ts_key = self._find_time_series_key(data)
        return self._parse_time_series(data[ts_key], start=start, end=end)

    def get_stock_intraday(
        self,
        symbol: str,
        interval: str = "5min",
        outputsize: str = "compact",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch intraday stock prices via ``TIME_SERIES_INTRADAY``."""
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }
        data = self._request(params)
        ts_key = self._find_time_series_key(data)
        return self._parse_time_series(data[ts_key], start=start, end=end)

    def get_crypto_daily(
        self,
        symbol: str,
        market: str = "USD",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch daily crypto prices via ``DIGITAL_CURRENCY_DAILY``."""
        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol,
            "market": market,
            "apikey": self.api_key,
        }
        data = self._request(params)
        ts_key = self._find_time_series_key(data)
        return self._parse_time_series(data[ts_key], start=start, end=end)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self, params: dict[str, str]) -> dict[str, Any]:
        """Execute a GET request and return the JSON payload."""
        resp = httpx.get(self.base_url, params=params, timeout=30.0)
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()

        if "Error Message" in payload:
            raise AlphaVantageError(payload["Error Message"])
        if "Note" in payload:
            raise AlphaVantageError(
                f"Alpha Vantage rate limit hit: {payload['Note']}"
            )
        if "Information" in payload and "Time Series" not in str(payload):
            raise AlphaVantageError(payload["Information"])

        return payload

    @staticmethod
    def _find_time_series_key(data: dict[str, Any]) -> str:
        """Find the key containing time-series data in the API response."""
        for key in data:
            if "Time Series" in key:
                return key
        raise AlphaVantageError(
            f"No time-series key found in response. Keys: {list(data.keys())}"
        )

    @staticmethod
    def _parse_time_series(
        raw: dict[str, dict[str, str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Convert the nested dict into a clean OHLCV DataFrame."""
        df = pd.DataFrame.from_dict(raw, orient="index")
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.sort_index()

        # Rename columns (handles both stock and crypto response shapes)
        df = df.rename(columns=_OHLCV_RENAME)

        # Keep only OHLCV columns
        ohlcv = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[ohlcv]

        # Cast to float
        for col in ohlcv:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Filter by date range if provided
        if start:
            df = df.loc[start:]  # type: ignore[misc]
        if end:
            df = df.loc[:end]  # type: ignore[misc]

        return df
