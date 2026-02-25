"""Tests for Signal and SignalType models."""

from trading_agent.models.signals import Signal, SignalType


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
