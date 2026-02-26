"""Tests for the Schwab broker stub."""

from __future__ import annotations

from trading_agent.execution.schwab_broker import SchwabBroker
from trading_agent.models.signals import Signal, SignalType


def test_schwab_stub_submit() -> None:
    broker = SchwabBroker()
    signal = Signal(signal_type=SignalType.BUY, symbol="AAPL", confidence=0.8)
    result = broker.submit_order(signal)
    assert result["status"] == "stub"


def test_schwab_stub_positions() -> None:
    broker = SchwabBroker()
    assert broker.get_positions() == []


def test_schwab_stub_balance() -> None:
    broker = SchwabBroker()
    assert broker.get_balance() == 0.0
