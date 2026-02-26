"""Tests for the risk management layer."""

from __future__ import annotations

import pytest

from trading_agent.execution.risk_manager import RiskConfig, RiskManager, RiskState
from trading_agent.models.signals import Order, OrderSide, OrderType


def make_buy_order(symbol: str = "AAPL", qty: float = 10.0) -> Order:
    return Order(symbol=symbol, side=OrderSide.BUY, qty=qty)


def make_sell_order(symbol: str = "AAPL", qty: float = 10.0) -> Order:
    return Order(symbol=symbol, side=OrderSide.SELL, qty=qty)


def test_approve_normal_order() -> None:
    rm = RiskManager()
    rm.update_state(current_equity=100_000, positions={}, total_exposure=0)
    result = rm.check_order(make_buy_order(), current_price=150.0)
    assert result.approved


def test_reject_when_halted() -> None:
    rm = RiskManager()
    rm.state.halted = True
    rm.state.halt_reason = "test halt"
    result = rm.check_order(make_buy_order(), current_price=150.0)
    assert not result.approved
    assert "halted" in result.reason.lower()


def test_reject_restricted_symbol() -> None:
    config = RiskConfig(restricted_symbols=["BANNED"])
    rm = RiskManager(config=config)
    rm.update_state(current_equity=100_000, positions={}, total_exposure=0)
    result = rm.check_order(make_buy_order(symbol="BANNED"), current_price=50.0)
    assert not result.approved
    assert "restricted" in result.reason.lower()


def test_reject_max_positions() -> None:
    config = RiskConfig(max_open_positions=2)
    rm = RiskManager(config=config)
    rm.update_state(
        current_equity=100_000,
        positions={"AAPL": 5000, "MSFT": 5000},
        total_exposure=10_000,
    )
    result = rm.check_order(make_buy_order(symbol="GOOGL"), current_price=100.0)
    assert not result.approved
    assert "positions" in result.reason.lower()


def test_reject_concentration() -> None:
    config = RiskConfig(max_position_pct=0.10)
    rm = RiskManager(config=config)
    rm.update_state(current_equity=100_000, positions={}, total_exposure=0)
    # 100 shares at $150 = $15,000 = 15% of equity
    result = rm.check_order(make_buy_order(qty=100), current_price=150.0)
    assert not result.approved
    assert "concentration" in result.reason.lower()


def test_reject_total_exposure() -> None:
    config = RiskConfig(max_total_exposure_pct=0.5)
    rm = RiskManager(config=config)
    rm.update_state(
        current_equity=100_000,
        positions={"MSFT": 45_000},
        total_exposure=45_000,
    )
    # Adding $10k would push to 55% > 50%
    result = rm.check_order(make_buy_order(qty=100), current_price=100.0)
    assert not result.approved
    assert "exposure" in result.reason.lower()


def test_sell_orders_bypass_position_limits() -> None:
    config = RiskConfig(max_open_positions=1)
    rm = RiskManager(config=config)
    rm.update_state(
        current_equity=100_000,
        positions={"AAPL": 5000},
        total_exposure=5000,
    )
    # Sell should be allowed even at max positions
    result = rm.check_order(make_sell_order(), current_price=150.0)
    assert result.approved


def test_drawdown_circuit_breaker() -> None:
    config = RiskConfig(max_drawdown_pct=0.10)
    rm = RiskManager(config=config)
    # Set peak at 100k
    rm.update_state(current_equity=100_000, positions={}, total_exposure=0)
    # Drop to 88k (12% drawdown > 10% limit)
    rm.update_state(current_equity=88_000, positions={}, total_exposure=0)
    assert rm.state.halted
    result = rm.check_order(make_buy_order(), current_price=100.0)
    assert not result.approved


def test_daily_loss_limit() -> None:
    config = RiskConfig(max_daily_loss_pct=0.02)
    rm = RiskManager(config=config)
    rm.start_new_day(equity=100_000)
    rm.update_state(current_equity=97_500, positions={}, total_exposure=0)
    # Daily loss = 2.5% > 2% limit
    result = rm.check_order(make_buy_order(), current_price=100.0)
    assert not result.approved
    assert "daily loss" in result.reason.lower()


def test_reset_halt() -> None:
    rm = RiskManager()
    rm.state.halted = True
    rm.reset_halt()
    assert not rm.state.halted
