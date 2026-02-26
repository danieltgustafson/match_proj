"""Tests for the alert system."""

from __future__ import annotations

from unittest.mock import patch

from trading_agent.alerts import AlertConfig, AlertManager


def test_trade_alert_logged() -> None:
    """Trade alerts should be logged even without channels configured."""
    manager = AlertManager(AlertConfig())
    # Should not raise
    manager.send_trade_alert("buy", "AAPL", 10, 195.50, "momentum", 0.75)


def test_risk_alert() -> None:
    manager = AlertManager(AlertConfig())
    manager.send_risk_alert("Daily loss limit hit: -3.5%")


def test_halt_alert() -> None:
    manager = AlertManager(AlertConfig())
    manager.send_halt_alert("Max drawdown breached: 16%")


def test_error_alert() -> None:
    manager = AlertManager(AlertConfig())
    manager.send_error_alert("Connection to IB lost")


def test_alerts_disabled() -> None:
    config = AlertConfig(
        alert_on_trade=False,
        alert_on_risk=False,
        alert_on_error=False,
    )
    manager = AlertManager(config)
    # These should be no-ops
    manager.send_trade_alert("buy", "AAPL", 10, 195.50, "test")
    manager.send_risk_alert("test")
    manager.send_error_alert("test")


@patch("trading_agent.alerts.httpx.post")
def test_slack_webhook(mock_post) -> None:
    mock_post.return_value.status_code = 200
    config = AlertConfig(slack_webhook="https://hooks.slack.com/test")
    manager = AlertManager(config)
    manager.send_trade_alert("buy", "AAPL", 10, 195.50, "momentum")
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert "AAPL" in call_args.kwargs.get("json", {}).get("text", "")


@patch("trading_agent.alerts.httpx.post")
def test_generic_webhook(mock_post) -> None:
    mock_post.return_value.status_code = 200
    config = AlertConfig(webhook_url="https://myserver.com/alerts")
    manager = AlertManager(config)
    manager.send_risk_alert("Drawdown warning")
    mock_post.assert_called_once()
