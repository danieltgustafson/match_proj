"""Tests for the factor-based fundamental strategy."""

from __future__ import annotations

from trading_agent.models.signals import SignalType
from trading_agent.strategies.factor import FactorStrategy, FundamentalData


def test_value_and_quality_buy() -> None:
    strategy = FactorStrategy(buy_threshold=0.05)
    fd = FundamentalData(
        symbol="CHEAP",
        pe_ratio=10.0,      # low P/E = good value
        gross_margin=0.6,   # high margin = good quality
        roe=0.25,           # high ROE
        momentum_12m=0.15,  # positive momentum
    )
    signal = strategy.evaluate_fundamentals(fd)
    assert signal.signal_type is SignalType.BUY
    assert signal.confidence > 0


def test_expensive_low_quality_sell() -> None:
    strategy = FactorStrategy(sell_threshold=-0.2)
    fd = FundamentalData(
        symbol="EXPENSIVE",
        pe_ratio=80.0,       # very high P/E
        gross_margin=0.05,   # thin margins
        roe=0.02,            # low ROE
        momentum_12m=-0.20,  # negative momentum
        volatility_daily=0.05,
    )
    signal = strategy.evaluate_fundamentals(fd)
    assert signal.signal_type is SignalType.SELL


def test_no_data_hold() -> None:
    strategy = FactorStrategy()
    fd = FundamentalData(symbol="NODATA")
    signal = strategy.evaluate_fundamentals(fd)
    assert signal.signal_type is SignalType.HOLD
    assert "insufficient data" in signal.metadata.get("reason", "")


def test_rank_universe() -> None:
    strategy = FactorStrategy()
    universe = [
        FundamentalData(symbol="GOOD", pe_ratio=12, gross_margin=0.5, momentum_12m=0.2),
        FundamentalData(symbol="MEH", pe_ratio=25, gross_margin=0.3, momentum_12m=0.05),
        FundamentalData(symbol="BAD", pe_ratio=60, gross_margin=0.1, momentum_12m=-0.15),
    ]
    signals = strategy.rank_universe(universe)
    assert len(signals) == 3
    # Best should be first
    symbols = [s.metadata.get("symbol", "") for s in signals]
    assert symbols[0] == "GOOD"
    assert symbols[-1] == "BAD"


def test_evaluate_from_dataframe() -> None:
    import pandas as pd
    strategy = FactorStrategy()
    df = pd.DataFrame({
        "close": [100.0, 105.0, 110.0, 115.0, 120.0] * 10,
        "pe_ratio": [15.0] * 50,
        "gross_margin": [0.4] * 50,
    })
    signal = strategy.evaluate(df)
    assert signal.signal_type in (SignalType.BUY, SignalType.SELL, SignalType.HOLD)
