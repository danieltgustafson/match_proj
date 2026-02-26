"""Alpha Vantage data provider for stocks and crypto.

Docs: https://www.alphavantage.co/documentation/

Supports:
- TIME_SERIES_DAILY / TIME_SERIES_INTRADAY for equities
- DIGITAL_CURRENCY_DAILY for crypto

Includes built-in rate limiting (1 req/sec), retry with exponential backoff,
and an in-memory cache to avoid burning through the free-tier quota
(25 requests/day).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import httpx
import pandas as pd

from trading_agent.data.providers.base import BaseProvider

logger = logging.getLogger("trading_agent.alpha_vantage")

_BASE_URL = "https://www.alphavantage.co/query"

# Maps Alpha Vantage column prefixes to clean names
_OHLCV_RENAME = {
    "1. open": "open",
    "2. high": "high",
    "3. low": "low",
    "4. close": "close",
    "5. volume": "volume",
}

# Strings that indicate a rate-limit / throttle response
_RATE_LIMIT_MARKERS = (
    "Thank you for using Alpha Vantage",
    "Please consider spreading out",
    "rate limit",
    "premium",
)


def _is_rate_limit_message(text: str) -> bool:
    """Return True if *text* looks like an AV rate-limit notice."""
    lower = text.lower()
    return any(marker.lower() in lower for marker in _RATE_LIMIT_MARKERS)


class AlphaVantageError(Exception):
    """Raised when the Alpha Vantage API returns an error or unexpected payload."""


class AlphaVantageRateLimitError(AlphaVantageError):
    """Raised specifically for rate-limit / quota responses."""


class AlphaVantageProvider(BaseProvider):
    """Fetch OHLCV data from the Alpha Vantage REST API.

    Parameters
    ----------
    api_key:
        Your Alpha Vantage API key.
    base_url:
        Override for testing / proxying.
    min_request_interval:
        Minimum seconds between consecutive HTTP requests (default 1.2s
        to stay safely under the 1-req/sec burst limit).
    max_retries:
        How many times to retry on rate-limit errors before giving up.
    cache_ttl:
        Seconds to keep cached responses (default 300 = 5 min).
        Set to 0 to disable caching.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _BASE_URL,
        min_request_interval: float = 1.2,
        max_retries: int = 3,
        cache_ttl: float = 300.0,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Alpha Vantage API key is required. "
                "Set ALPHA_VANTAGE_API_KEY in your .env file."
            )
        self.api_key = api_key
        self.base_url = base_url
        self.min_request_interval = min_request_interval
        self.max_retries = max_retries
        self.cache_ttl = cache_ttl

        self._last_request_time: float = 0.0
        # Cache: key -> (timestamp, payload)
        self._cache: Dict[str, Tuple[float, dict]] = {}

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
        data = self._request_with_retry(params)
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
        data = self._request_with_retry(params)
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
        data = self._request_with_retry(params)
        ts_key = self._find_time_series_key(data)
        return self._parse_time_series(data[ts_key], start=start, end=end)

    def clear_cache(self) -> None:
        """Drop all cached responses."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Rate limiting, caching, and retry
    # ------------------------------------------------------------------

    def _cache_key(self, params: dict[str, str]) -> str:
        """Build a deterministic cache key from request params (excluding apikey)."""
        filtered = {k: v for k, v in sorted(params.items()) if k != "apikey"}
        return "&".join(f"{k}={v}" for k, v in filtered.items())

    def _get_cached(self, key: str) -> Optional[dict]:
        """Return cached payload if it exists and hasn't expired."""
        if self.cache_ttl <= 0:
            return None
        entry = self._cache.get(key)
        if entry is None:
            return None
        cached_at, payload = entry
        if (time.monotonic() - cached_at) > self.cache_ttl:
            del self._cache[key]
            return None
        return payload

    def _set_cached(self, key: str, payload: dict) -> None:
        if self.cache_ttl > 0:
            self._cache[key] = (time.monotonic(), payload)

    def _throttle(self) -> None:
        """Sleep if needed to respect the minimum request interval."""
        if self.min_request_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_time
        wait = self.min_request_interval - elapsed
        if wait > 0:
            logger.debug("Throttling for %.2fs before next AV request", wait)
            time.sleep(wait)

    def _request_with_retry(self, params: dict[str, str]) -> dict[str, Any]:
        """Execute a request with caching, throttling, and retry on rate limits."""
        cache_key = self._cache_key(params)
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug("Cache hit for %s", cache_key)
            return cached

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                payload = self._request(params)
                self._set_cached(cache_key, payload)
                return payload
            except AlphaVantageRateLimitError as exc:
                last_err = exc
                backoff = 2 ** attempt  # 2s, 4s, 8s ...
                logger.warning(
                    "Rate limited on attempt %d/%d for %s. "
                    "Retrying in %ds...",
                    attempt,
                    self.max_retries,
                    params.get("symbol", "?"),
                    backoff,
                )
                time.sleep(backoff)

        raise AlphaVantageRateLimitError(
            f"Still rate-limited after {self.max_retries} retries. "
            f"Last error: {last_err}"
        )

    def _request(self, params: dict[str, str]) -> dict[str, Any]:
        """Execute a single throttled GET request and return the JSON payload."""
        self._throttle()
        self._last_request_time = time.monotonic()

        resp = httpx.get(self.base_url, params=params, timeout=30.0)
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()

        # Check for error conditions
        if "Error Message" in payload:
            raise AlphaVantageError(payload["Error Message"])

        # Rate-limit responses come in "Note" or "Information" keys
        if "Note" in payload and _is_rate_limit_message(payload["Note"]):
            raise AlphaVantageRateLimitError(payload["Note"])

        if "Information" in payload and _is_rate_limit_message(payload["Information"]):
            raise AlphaVantageRateLimitError(payload["Information"])

        # Non-rate-limit "Information" (e.g. genuinely invalid symbol)
        if "Information" in payload and "Time Series" not in str(payload):
            raise AlphaVantageError(payload["Information"])

        return payload

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

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
