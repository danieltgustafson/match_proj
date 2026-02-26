"""Tests for the SQLite trade store."""

from __future__ import annotations

import os
import tempfile

import pytest

from trading_agent.storage import TradeStore


@pytest.fixture()
def store() -> TradeStore:
    """Create a temporary in-memory-like store for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = TradeStore(path)
    yield s
    s.close()
    os.unlink(path)


def test_log_and_get_trade(store: TradeStore) -> None:
    trade_id = store.log_trade(
        symbol="AAPL", side="buy", qty=10, price=195.50,
        strategy="momentum", status="filled", confidence=0.75,
    )
    assert trade_id > 0
    trades = store.get_trades()
    assert len(trades) == 1
    assert trades[0]["symbol"] == "AAPL"
    assert trades[0]["qty"] == 10


def test_log_multiple_trades(store: TradeStore) -> None:
    store.log_trade(symbol="AAPL", side="buy", qty=10, price=195.0)
    store.log_trade(symbol="MSFT", side="sell", qty=5, price=420.0)
    store.log_trade(symbol="AAPL", side="sell", qty=10, price=200.0)

    all_trades = store.get_trades()
    assert len(all_trades) == 3

    aapl_trades = store.get_trades(symbol="AAPL")
    assert len(aapl_trades) == 2


def test_log_signal(store: TradeStore) -> None:
    store.log_signal(symbol="NVDA", signal_type="buy", confidence=0.9, strategy="ensemble")
    signals = store.get_signals()
    assert len(signals) == 1
    assert signals[0]["symbol"] == "NVDA"


def test_portfolio_snapshot(store: TradeStore) -> None:
    store.log_portfolio_snapshot(
        equity=100_000, cash=50_000,
        positions=[{"symbol": "AAPL", "market_value": 25_000}],
    )
    history = store.get_portfolio_history()
    assert len(history) == 1
    assert history[0]["equity"] == 100_000
    assert history[0]["num_positions"] == 1


def test_alerts(store: TradeStore) -> None:
    store.log_alert("risk", "Daily loss limit hit: -3.5%")
    store.log_alert("trade", "BUY AAPL x10 @ $195")

    alerts = store.get_alerts(unacknowledged_only=True)
    assert len(alerts) == 2

    store.acknowledge_alert(alerts[0]["id"])
    unacked = store.get_alerts(unacknowledged_only=True)
    assert len(unacked) == 1


def test_trade_summary(store: TradeStore) -> None:
    store.log_trade(symbol="AAPL", side="buy", qty=10, price=195.0, confidence=0.8)
    store.log_trade(symbol="MSFT", side="sell", qty=5, price=420.0, confidence=0.6)

    summary = store.get_trade_summary()
    assert summary["total_trades"] == 2
    assert summary["buys"] == 1
    assert summary["sells"] == 1
    assert summary["unique_symbols"] == 2


def test_scanner_results(store: TradeStore) -> None:
    store.log_scanner_result(
        num_scanned=4000, num_passed=847,
        top_symbols=["NVDA", "SMCI", "MARA"],
    )
    # Just verify it doesn't crash -- no query method yet for scanner results
