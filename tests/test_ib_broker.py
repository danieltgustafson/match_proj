"""Tests for the Interactive Brokers broker integration.

These tests validate the IBBroker class structure and error handling
without requiring a live TWS/IB Gateway connection.
"""

from __future__ import annotations

import pytest

from trading_agent.execution.ib_broker import IBBroker, IBConnectionError
from trading_agent.models.signals import OrderStatus, Signal, SignalType


def test_ib_broker_instantiation() -> None:
    broker = IBBroker(host="127.0.0.1", port=7497, client_id=1)
    assert broker.host == "127.0.0.1"
    assert broker.port == 7497
    assert broker.client_id == 1
    assert not broker._connected


def test_ib_broker_requires_connection() -> None:
    broker = IBBroker()
    with pytest.raises(IBConnectionError, match="Not connected"):
        broker.get_account()


def test_ib_broker_get_positions_requires_connection() -> None:
    broker = IBBroker()
    with pytest.raises(IBConnectionError, match="Not connected"):
        broker.list_positions()


def test_ib_broker_submit_hold_skips() -> None:
    """HOLD signals should be skipped without needing a connection."""
    broker = IBBroker()
    signal = Signal(signal_type=SignalType.HOLD)
    result = broker.submit_order(signal)
    assert result["status"] == "skipped"


def test_ib_status_mapping() -> None:
    assert IBBroker._map_ib_status("Filled") is OrderStatus.FILLED
    assert IBBroker._map_ib_status("Submitted") is OrderStatus.SUBMITTED
    assert IBBroker._map_ib_status("Cancelled") is OrderStatus.CANCELLED
    assert IBBroker._map_ib_status("Unknown") is OrderStatus.PENDING
