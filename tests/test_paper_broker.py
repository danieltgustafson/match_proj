"""Tests for the paper trading broker."""

from trading_agent.execution.paper_trading import PaperBroker
from trading_agent.models.signals import Signal, SignalType


def test_initial_balance() -> None:
    broker = PaperBroker(initial_cash=50_000.0)
    assert broker.get_balance() == 50_000.0


def test_submit_actionable_order() -> None:
    broker = PaperBroker()
    signal = Signal(signal_type=SignalType.BUY, symbol="AAPL", confidence=0.9)
    result = broker.submit_order(signal)
    assert result["status"] == "filled"
    assert len(broker.order_log) == 1


def test_submit_hold_skipped() -> None:
    broker = PaperBroker()
    signal = Signal(signal_type=SignalType.HOLD, confidence=0.0)
    result = broker.submit_order(signal)
    assert result["status"] == "skipped"


def test_get_positions_empty() -> None:
    broker = PaperBroker()
    assert broker.get_positions() == []


def test_get_account() -> None:
    broker = PaperBroker(initial_cash=75_000.0)
    account = broker.get_account()
    assert account["equity"] == 75_000.0
    assert account["cash"] == 75_000.0
