"""Tests for the universe scanner."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from trading_agent.scanner import (
    ALL_SEEDS,
    CRYPTO_RELATED,
    GROWTH_TECH,
    SP500_SAMPLE,
    ScannerConfig,
    UniverseScanner,
)


def test_seed_universe_all() -> None:
    scanner = UniverseScanner(ScannerConfig(universe="all"))
    seeds = scanner._get_seed_universe()
    assert len(seeds) > 50


def test_seed_universe_sp500() -> None:
    scanner = UniverseScanner(ScannerConfig(universe="sp500"))
    seeds = scanner._get_seed_universe()
    assert "AAPL" in seeds
    assert "MSFT" in seeds


def test_seed_universe_growth() -> None:
    scanner = UniverseScanner(ScannerConfig(universe="growth"))
    seeds = scanner._get_seed_universe()
    assert "PLTR" in seeds


def test_seed_universe_crypto() -> None:
    scanner = UniverseScanner(ScannerConfig(universe="crypto"))
    seeds = scanner._get_seed_universe()
    assert "COIN" in seeds
    assert "MARA" in seeds


def test_custom_symbols_override() -> None:
    config = ScannerConfig(custom_symbols=["FOO", "BAR"])
    scanner = UniverseScanner(config)
    seeds = scanner._get_seed_universe()
    assert seeds == ["FOO", "BAR"]


def test_apply_filters() -> None:
    scanner = UniverseScanner(ScannerConfig(
        min_avg_volume=100_000,
        min_price=10.0,
        max_price=500.0,
        min_dollar_volume=1_000_000,
        min_volatility=0.005,
        max_volatility=0.10,
    ))
    df = pd.DataFrame({
        "last_price": [150.0, 3.0, 200.0, 50.0],
        "avg_volume": [500_000, 200_000, 1_000_000, 50_000],
        "avg_dollar_volume": [75_000_000, 600_000, 200_000_000, 2_500_000],
        "momentum_20d": [0.05, 0.10, -0.02, 0.30],
        "volatility": [0.02, 0.03, 0.015, 0.008],
        "trend_strength": [1.5, 0.8, 0.5, 2.0],
    }, index=["AAPL", "PENNY", "MSFT", "LOWVOL"])

    filtered = scanner._apply_filters(df)
    assert "AAPL" in filtered.index
    assert "MSFT" in filtered.index
    assert "PENNY" not in filtered.index   # price too low
    assert "LOWVOL" not in filtered.index  # volume too low


def test_rank_composite() -> None:
    scanner = UniverseScanner(ScannerConfig(rank_by="composite"))
    df = pd.DataFrame({
        "last_price": [150.0, 200.0, 50.0],
        "avg_volume": [500_000, 1_000_000, 2_000_000],
        "avg_dollar_volume": [75_000_000, 200_000_000, 100_000_000],
        "momentum_20d": [0.10, -0.05, 0.20],
        "volatility": [0.02, 0.015, 0.025],
        "trend_strength": [2.0, 0.5, 3.0],
    }, index=["A", "B", "C"])

    ranked = scanner._rank(df)
    assert ranked.index[0] == "C"  # highest momentum + trend


def test_rank_momentum() -> None:
    scanner = UniverseScanner(ScannerConfig(rank_by="momentum"))
    df = pd.DataFrame({
        "momentum_20d": [0.05, 0.20, 0.10],
        "last_price": [100, 100, 100],
        "avg_volume": [1e6, 1e6, 1e6],
        "avg_dollar_volume": [1e8, 1e8, 1e8],
        "volatility": [0.02, 0.02, 0.02],
        "trend_strength": [1, 1, 1],
    }, index=["A", "B", "C"])

    ranked = scanner._rank(df)
    assert ranked.index[0] == "B"  # highest momentum


def test_us_all_fetches_dynamic_listings() -> None:
    """Test that us_all mode calls Alpha Vantage LISTING_STATUS."""
    csv_data = (
        "symbol,name,exchange,assetType,ipoDate,delistingDate,status\n"
        "AAPL,Apple Inc,NYSE,Stock,1980-12-12,,Active\n"
        "MSFT,Microsoft Corp,NASDAQ,Stock,1986-03-13,,Active\n"
        "TSLA,Tesla Inc,NASDAQ,Stock,2010-06-29,,Active\n"
        "SPYETF,SPDR ETF,NYSE,ETF,1993-01-22,,Active\n"
        "DEAD,Dead Corp,NYSE,Stock,2000-01-01,2023-06-01,Delisted\n"
    )

    class FakeResp:
        status_code = 200
        text = csv_data

    config = ScannerConfig(universe="us_all", alpha_vantage_key="test-key")
    scanner = UniverseScanner(config)

    with patch("trading_agent.scanner.httpx.get", return_value=FakeResp()):
        symbols = scanner._fetch_us_listings()

    # Should include active stocks, exclude ETFs and delisted
    assert "AAPL" in symbols
    assert "MSFT" in symbols
    assert "TSLA" in symbols
    assert "DEAD" not in symbols  # delisted


def test_us_all_fallback_without_key() -> None:
    """Without an API key, us_all should fall back to static seeds."""
    config = ScannerConfig(universe="us_all", alpha_vantage_key="")
    scanner = UniverseScanner(config)
    seeds = scanner._get_seed_universe()
    # Should get the static ALL_SEEDS fallback
    assert len(seeds) > 50


def test_scan_returns_list() -> None:
    """Integration test with mocked yfinance data."""
    scanner = UniverseScanner(ScannerConfig(
        custom_symbols=["AAPL", "MSFT"],
        max_symbols=2,
    ))

    # Mock yfinance.download where it's imported inside the method
    fake_data = pd.DataFrame({
        ("AAPL", "Close"): np.linspace(150, 160, 20),
        ("AAPL", "Volume"): [1_000_000.0] * 20,
        ("MSFT", "Close"): np.linspace(300, 310, 20),
        ("MSFT", "Volume"): [2_000_000.0] * 20,
    })
    fake_data.columns = pd.MultiIndex.from_tuples(fake_data.columns)

    with patch("yfinance.download", return_value=fake_data):
        result = scanner.scan()

    assert isinstance(result, list)
    assert len(result) <= 2
