"""Tests for the Alpha Vantage data provider."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from trading_agent.data.providers.alpha_vantage import (
    AlphaVantageError,
    AlphaVantageProvider,
    AlphaVantageRateLimitError,
)

# ---------------------------------------------------------------------------
# Fixtures: fake API responses
# ---------------------------------------------------------------------------

FAKE_STOCK_DAILY: dict[str, Any] = {
    "Meta Data": {
        "1. Information": "Daily Prices (open, high, low, close) and Volumes",
        "2. Symbol": "AAPL",
        "3. Last Refreshed": "2024-06-01",
    },
    "Time Series (Daily)": {
        "2024-06-01": {
            "1. open": "190.00",
            "2. high": "195.00",
            "3. low": "189.00",
            "4. close": "194.50",
            "5. volume": "50000000",
        },
        "2024-05-31": {
            "1. open": "188.00",
            "2. high": "191.00",
            "3. low": "187.00",
            "4. close": "190.00",
            "5. volume": "45000000",
        },
    },
}

FAKE_CRYPTO_DAILY: dict[str, Any] = {
    "Meta Data": {
        "1. Information": "Daily Prices and Volumes for Digital Currency",
        "2. Digital Currency Code": "BTC",
        "3. Digital Currency Name": "Bitcoin",
    },
    "Time Series (Digital Currency Daily)": {
        "2024-06-01": {
            "1. open": "67000.00",
            "2. high": "68500.00",
            "3. low": "66500.00",
            "4. close": "68000.00",
            "5. volume": "1500.12",
        },
        "2024-05-31": {
            "1. open": "66000.00",
            "2. high": "67200.00",
            "3. low": "65800.00",
            "4. close": "67000.00",
            "5. volume": "1300.50",
        },
    },
}

RATE_LIMIT_NOTE = {
    "Note": (
        "Thank you for using Alpha Vantage! Please consider spreading out "
        "your free API requests more sparingly."
    )
}

RATE_LIMIT_INFO = {
    "Information": (
        "Thank you for using Alpha Vantage! Please consider spreading out "
        "your free API requests more sparingly (1 request per second)."
    )
}


class FakeResponse:
    """Minimal httpx.Response stand-in."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


def _make_provider(**overrides: Any) -> AlphaVantageProvider:
    """Create a provider with throttling disabled for fast tests."""
    defaults = {
        "api_key": "test-key",
        "min_request_interval": 0,  # no throttle in tests
        "max_retries": 3,
        "cache_ttl": 300,
    }
    defaults.update(overrides)
    return AlphaVantageProvider(**defaults)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


def test_requires_api_key() -> None:
    with pytest.raises(ValueError, match="API key is required"):
        AlphaVantageProvider(api_key="")


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_get_stock_daily(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse(FAKE_STOCK_DAILY)
    provider = _make_provider()
    df = provider.get_stock_daily("AAPL")

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df["close"].iloc[-1] == 194.50
    assert df.index[0] < df.index[1]


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_get_crypto_daily(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse(FAKE_CRYPTO_DAILY)
    provider = _make_provider()
    df = provider.get_crypto_daily("BTC")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df["close"].iloc[-1] == 68000.00


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_get_historical_stock(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse(FAKE_STOCK_DAILY)
    provider = _make_provider()
    df = provider.get_historical("AAPL")
    assert len(df) == 2


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_get_historical_crypto(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse(FAKE_CRYPTO_DAILY)
    provider = _make_provider()
    df = provider.get_historical("BTC", asset_type="crypto")
    assert len(df) == 2


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_date_filtering(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse(FAKE_STOCK_DAILY)
    provider = _make_provider()
    df = provider.get_stock_daily("AAPL", start="2024-06-01", end="2024-06-01")
    assert len(df) == 1
    assert str(df.index[0].date()) == "2024-06-01"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_error_message_raises(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse({"Error Message": "Invalid API call"})
    provider = _make_provider(max_retries=1)
    with pytest.raises(AlphaVantageError, match="Invalid API call"):
        provider.get_stock_daily("INVALID")


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_rate_limit_note_raises(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse(RATE_LIMIT_NOTE)
    provider = _make_provider(max_retries=1)
    with pytest.raises(AlphaVantageRateLimitError):
        provider.get_stock_daily("AAPL")


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_rate_limit_information_raises(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse(RATE_LIMIT_INFO)
    provider = _make_provider(max_retries=1)
    with pytest.raises(AlphaVantageRateLimitError):
        provider.get_stock_daily("AAPL")


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


@patch("trading_agent.data.providers.alpha_vantage.time.sleep")
@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_retries_on_rate_limit_then_succeeds(mock_get: Any, mock_sleep: Any) -> None:
    """First call hits rate limit, second call succeeds."""
    mock_get.side_effect = [
        FakeResponse(RATE_LIMIT_INFO),
        FakeResponse(FAKE_STOCK_DAILY),
    ]
    provider = _make_provider(max_retries=3)
    df = provider.get_stock_daily("AAPL")

    assert len(df) == 2
    # Should have called httpx.get twice
    assert mock_get.call_count == 2
    # Should have slept once for backoff (2^1 = 2 seconds)
    mock_sleep.assert_called()


@patch("trading_agent.data.providers.alpha_vantage.time.sleep")
@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_exhausts_retries_raises(mock_get: Any, mock_sleep: Any) -> None:
    """All retries hit rate limit -> raises AlphaVantageRateLimitError."""
    mock_get.return_value = FakeResponse(RATE_LIMIT_INFO)
    provider = _make_provider(max_retries=2)

    with pytest.raises(AlphaVantageRateLimitError, match="Still rate-limited"):
        provider.get_stock_daily("AAPL")

    assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_cache_avoids_duplicate_requests(mock_get: Any) -> None:
    """Second call for the same symbol should come from cache."""
    mock_get.return_value = FakeResponse(FAKE_STOCK_DAILY)
    provider = _make_provider(cache_ttl=300)

    df1 = provider.get_stock_daily("AAPL")
    df2 = provider.get_stock_daily("AAPL")

    # Only one HTTP request should have been made
    assert mock_get.call_count == 1
    assert len(df1) == len(df2) == 2


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_cache_disabled_when_ttl_zero(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse(FAKE_STOCK_DAILY)
    provider = _make_provider(cache_ttl=0)

    provider.get_stock_daily("AAPL")
    provider.get_stock_daily("AAPL")

    assert mock_get.call_count == 2


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_clear_cache(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse(FAKE_STOCK_DAILY)
    provider = _make_provider(cache_ttl=300)

    provider.get_stock_daily("AAPL")
    provider.clear_cache()
    provider.get_stock_daily("AAPL")

    assert mock_get.call_count == 2


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_different_symbols_not_cached_together(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse(FAKE_STOCK_DAILY)
    provider = _make_provider(cache_ttl=300)

    provider.get_stock_daily("AAPL")
    provider.get_stock_daily("MSFT")

    assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Throttle
# ---------------------------------------------------------------------------


@patch("trading_agent.data.providers.alpha_vantage.time.sleep")
@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_throttle_sleeps_between_requests(mock_get: Any, mock_sleep: Any) -> None:
    """With throttle enabled and cache disabled, consecutive calls should sleep."""
    mock_get.return_value = FakeResponse(FAKE_STOCK_DAILY)
    provider = _make_provider(min_request_interval=1.2, cache_ttl=0)

    provider.get_stock_daily("AAPL")
    provider.get_stock_daily("AAPL")

    # time.sleep should have been called for throttling on the second request
    assert mock_sleep.call_count >= 1


# ---------------------------------------------------------------------------
# Disk cache (L2) integration
# ---------------------------------------------------------------------------


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_disk_cache_persists_across_provider_instances(
    mock_get: Any, tmp_path: Any
) -> None:
    """Data fetched by one provider instance is available to a new one via disk."""
    db_path = str(tmp_path / "av_cache.db")
    mock_get.return_value = FakeResponse(FAKE_STOCK_DAILY)

    # First provider fetches and caches to disk
    p1 = _make_provider(disk_cache_path=db_path, disk_cache_ttl=3600)
    df1 = p1.get_stock_daily("AAPL")
    assert mock_get.call_count == 1

    # Second provider (simulating a restart) should get disk hit
    p2 = _make_provider(disk_cache_path=db_path, disk_cache_ttl=3600)
    df2 = p2.get_stock_daily("AAPL")

    # No additional HTTP call -- served from disk
    assert mock_get.call_count == 1
    assert len(df2) == len(df1)


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_disk_cache_disabled_by_default(mock_get: Any) -> None:
    """With default settings (no disk_cache_path), disk cache is not used."""
    mock_get.return_value = FakeResponse(FAKE_STOCK_DAILY)
    provider = _make_provider()  # disk_cache_path="" by default
    assert provider._disk_cache is None

    provider.get_stock_daily("AAPL")
    provider._cache.clear()  # clear in-memory
    provider.get_stock_daily("AAPL")

    # Should have made 2 HTTP calls since there's no disk fallback
    assert mock_get.call_count == 2
