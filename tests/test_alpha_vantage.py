"""Tests for the Alpha Vantage data provider."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from trading_agent.data.providers.alpha_vantage import (
    AlphaVantageError,
    AlphaVantageProvider,
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_requires_api_key() -> None:
    with pytest.raises(ValueError, match="API key is required"):
        AlphaVantageProvider(api_key="")


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_get_stock_daily(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse(FAKE_STOCK_DAILY)
    provider = AlphaVantageProvider(api_key="test-key")
    df = provider.get_stock_daily("AAPL")

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df["close"].iloc[-1] == 194.50
    # Verify sorted ascending
    assert df.index[0] < df.index[1]


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_get_crypto_daily(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse(FAKE_CRYPTO_DAILY)
    provider = AlphaVantageProvider(api_key="test-key")
    df = provider.get_crypto_daily("BTC")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df["close"].iloc[-1] == 68000.00


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_get_historical_stock(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse(FAKE_STOCK_DAILY)
    provider = AlphaVantageProvider(api_key="test-key")
    df = provider.get_historical("AAPL")
    assert len(df) == 2


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_get_historical_crypto(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse(FAKE_CRYPTO_DAILY)
    provider = AlphaVantageProvider(api_key="test-key")
    df = provider.get_historical("BTC", asset_type="crypto")
    assert len(df) == 2


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_date_filtering(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse(FAKE_STOCK_DAILY)
    provider = AlphaVantageProvider(api_key="test-key")
    df = provider.get_stock_daily("AAPL", start="2024-06-01", end="2024-06-01")
    assert len(df) == 1
    assert str(df.index[0].date()) == "2024-06-01"


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_error_message_raises(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse({"Error Message": "Invalid API call"})
    provider = AlphaVantageProvider(api_key="test-key")
    with pytest.raises(AlphaVantageError, match="Invalid API call"):
        provider.get_stock_daily("INVALID")


@patch("trading_agent.data.providers.alpha_vantage.httpx.get")
def test_rate_limit_raises(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse(
        {"Note": "Thank you for using Alpha Vantage! Please limit to 25 requests per day."}
    )
    provider = AlphaVantageProvider(api_key="test-key")
    with pytest.raises(AlphaVantageError, match="rate limit"):
        provider.get_stock_daily("AAPL")
