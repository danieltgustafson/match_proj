"""Tests for Signal, Order, and Position models."""

from trading_agent.models.signals import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Signal,
    SignalType,
    TimeInForce,
)


def test_signal_hold_not_actionable() -> None:
    sig = Signal(signal_type=SignalType.HOLD)
    assert not sig.is_actionable


def test_signal_buy_is_actionable() -> None:
    sig = Signal(signal_type=SignalType.BUY, confidence=0.8)
    assert sig.is_actionable


def test_signal_sell_is_actionable() -> None:
    sig = Signal(signal_type=SignalType.SELL, confidence=0.5)
    assert sig.is_actionable


def test_signal_is_frozen() -> None:
    sig = Signal(signal_type=SignalType.BUY)
    try:
        sig.confidence = 1.0  # type: ignore[misc]
        raise AssertionError("Should not be able to mutate a frozen dataclass")
    except AttributeError:
        pass


def test_signal_with_symbol() -> None:
    sig = Signal(signal_type=SignalType.BUY, symbol="AAPL", strategy_name="momentum")
    assert sig.symbol == "AAPL"
    assert sig.strategy_name == "momentum"


def test_order_to_dict() -> None:
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=10.0,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
    )
    d = order.to_dict()
    assert d["symbol"] == "AAPL"
    assert d["side"] == "buy"
    assert d["qty"] == 10.0
    assert d["status"] == "pending"


def test_position_long_short() -> None:
    long = Position(symbol="AAPL", qty=10, avg_entry_price=150.0)
    assert long.is_long
    assert not long.is_short

    short = Position(symbol="TSLA", qty=-5, avg_entry_price=200.0)
    assert short.is_short
    assert not short.is_long
