"""Tests for the portfolio manager and position sizing."""

from __future__ import annotations

import pytest

from trading_agent.execution.portfolio_manager import PortfolioConfig, PortfolioManager
from trading_agent.models.signals import Signal, SignalType


@pytest.fixture()
def pm() -> PortfolioManager:
    config = PortfolioConfig(
        max_position_pct=0.10,
        sizing_method="kelly",
        kelly_fraction=0.5,
        min_order_value=100.0,
    )
    return PortfolioManager(equity=100_000, cash=50_000, config=config)


def test_hold_signal_returns_none(pm: PortfolioManager) -> None:
    sig = Signal(signal_type=SignalType.HOLD, symbol="AAPL")
    assert pm.signal_to_order(sig, current_price=150.0) is None


def test_buy_signal_produces_order(pm: PortfolioManager) -> None:
    sig = Signal(signal_type=SignalType.BUY, symbol="AAPL", confidence=0.8)
    order = pm.signal_to_order(sig, current_price=150.0)
    assert order is not None
    assert order.symbol == "AAPL"
    assert order.side.value == "buy"
    assert order.qty >= 1


def test_sell_without_position_returns_none(pm: PortfolioManager) -> None:
    sig = Signal(signal_type=SignalType.SELL, symbol="AAPL", confidence=0.9)
    order = pm.signal_to_order(sig, current_price=150.0)
    assert order is None


def test_sell_with_position_produces_order() -> None:
    pm = PortfolioManager(
        equity=100_000,
        cash=50_000,
        current_positions={"AAPL": 50.0},
    )
    sig = Signal(signal_type=SignalType.SELL, symbol="AAPL", confidence=0.9)
    order = pm.signal_to_order(sig, current_price=150.0)
    assert order is not None
    assert order.side.value == "sell"
    assert order.qty <= 50


def test_kelly_sizing_positive() -> None:
    pm = PortfolioManager(equity=100_000, cash=100_000)
    # With favorable odds, Kelly should produce a positive fraction
    order = pm.signal_to_order(
        Signal(signal_type=SignalType.BUY, symbol="TEST", confidence=0.7),
        current_price=50.0,
        win_rate=0.6,
        avg_win=2.0,
        avg_loss=1.0,
    )
    assert order is not None
    assert order.qty > 0


def test_risk_parity_sizing() -> None:
    config = PortfolioConfig(sizing_method="risk_parity")
    pm = PortfolioManager(equity=100_000, cash=100_000, config=config)
    order = pm.signal_to_order(
        Signal(signal_type=SignalType.BUY, symbol="TEST", confidence=0.6),
        current_price=50.0,
        volatility=0.02,
    )
    assert order is not None
    assert order.qty > 0


def test_fixed_fractional_sizing() -> None:
    config = PortfolioConfig(sizing_method="fixed_fractional", fixed_fraction=0.05)
    pm = PortfolioManager(equity=100_000, cash=100_000, config=config)
    order = pm.signal_to_order(
        Signal(signal_type=SignalType.BUY, symbol="TEST", confidence=0.8),
        current_price=100.0,
    )
    assert order is not None
    # 5% of 100k * 0.8 confidence = $4000 / $100 = 40 shares
    assert order.qty == 40


def test_zero_price_returns_none(pm: PortfolioManager) -> None:
    sig = Signal(signal_type=SignalType.BUY, symbol="AAPL", confidence=0.5)
    assert pm.signal_to_order(sig, current_price=0.0) is None
