"""Alert system -- notifications via Slack, email, or webhook.

Sends alerts when:
- Trades execute
- Risk limits are triggered (drawdown, daily loss)
- Trading is halted
- Scanner finds unusual activity
- Errors occur

Supports multiple channels:
- Slack (via webhook URL)
- Email (via SMTP)
- Generic webhook (POST to any URL)
- Console (always on, for logging)

Usage::

    from trading_agent.alerts import AlertManager

    alerts = AlertManager(slack_webhook="https://hooks.slack.com/...")
    alerts.send_trade_alert("BUY", "AAPL", 10, 195.50, "momentum")
    alerts.send_risk_alert("Daily loss limit hit: -3.2%")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger("trading_agent.alerts")


@dataclass
class AlertConfig:
    """Alert channel configuration."""

    # Slack webhook URL (get from Slack App settings)
    slack_webhook: str = ""

    # Email settings
    email_enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_to: str = ""
    email_from: str = ""

    # Generic webhook (POST JSON to any URL)
    webhook_url: str = ""

    # Which events to alert on
    alert_on_trade: bool = True
    alert_on_risk: bool = True
    alert_on_error: bool = True
    alert_on_halt: bool = True
    alert_on_scanner: bool = False  # can be noisy


class AlertManager:
    """Send alerts to configured channels."""

    def __init__(self, config: Optional[AlertConfig] = None) -> None:
        self.config = config or AlertConfig()

    def send_trade_alert(
        self,
        side: str,
        symbol: str,
        qty: float,
        price: float,
        strategy: str,
        confidence: float = 0.0,
    ) -> None:
        """Notify about a trade execution."""
        if not self.config.alert_on_trade:
            return
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        msg = (
            f"{emoji} **{side.upper()} {symbol}** x{qty:.0f} @ ${price:.2f}\n"
            f"Strategy: {strategy} | Confidence: {confidence:.0%}"
        )
        self._send("trade", msg)

    def send_risk_alert(self, message: str) -> None:
        """Notify about a risk event (daily loss, drawdown, etc.)."""
        if not self.config.alert_on_risk:
            return
        self._send("risk", f"⚠️ **Risk Alert**: {message}")

    def send_halt_alert(self, reason: str) -> None:
        """Notify that trading has been halted."""
        if not self.config.alert_on_halt:
            return
        self._send("halt", f"🛑 **TRADING HALTED**: {reason}")

    def send_error_alert(self, error: str) -> None:
        """Notify about an error."""
        if not self.config.alert_on_error:
            return
        self._send("error", f"❌ **Error**: {error}")

    def send_scanner_alert(self, top_symbols: list[str], num_total: int) -> None:
        """Notify about scanner results."""
        if not self.config.alert_on_scanner:
            return
        self._send(
            "scanner",
            f"🔍 Scanner found {num_total} candidates. "
            f"Top picks: {', '.join(top_symbols[:10])}",
        )

    def send_daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        num_trades: int,
        top_winners: list[str],
        top_losers: list[str],
    ) -> None:
        """Send end-of-day summary."""
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        msg = (
            f"{pnl_emoji} **Daily Summary**\n"
            f"Equity: ${equity:,.2f}\n"
            f"P&L: ${daily_pnl:,.2f} ({daily_pnl_pct:+.2%})\n"
            f"Trades: {num_trades}\n"
        )
        if top_winners:
            msg += f"Winners: {', '.join(top_winners)}\n"
        if top_losers:
            msg += f"Losers: {', '.join(top_losers)}\n"
        self._send("summary", msg)

    # ------------------------------------------------------------------
    # Channel implementations
    # ------------------------------------------------------------------

    def _send(self, alert_type: str, message: str) -> None:
        """Route message to all configured channels."""
        logger.info("[ALERT:%s] %s", alert_type, message)

        if self.config.slack_webhook:
            self._send_slack(message)

        if self.config.webhook_url:
            self._send_webhook(alert_type, message)

        if self.config.email_enabled:
            self._send_email(alert_type, message)

    def _send_slack(self, message: str) -> None:
        """Post to Slack via incoming webhook."""
        try:
            # Convert markdown bold to Slack bold
            slack_msg = message.replace("**", "*")
            resp = httpx.post(
                self.config.slack_webhook,
                json={"text": slack_msg},
                timeout=10.0,
            )
            if resp.status_code != 200:
                logger.warning("Slack alert failed: %s", resp.text)
        except Exception as exc:
            logger.warning("Slack alert error: %s", exc)

    def _send_webhook(self, alert_type: str, message: str) -> None:
        """POST JSON to a generic webhook URL."""
        try:
            payload = {
                "type": alert_type,
                "message": message,
                "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
            }
            resp = httpx.post(
                self.config.webhook_url,
                json=payload,
                timeout=10.0,
            )
            if resp.status_code >= 400:
                logger.warning("Webhook alert failed: %s", resp.text)
        except Exception as exc:
            logger.warning("Webhook alert error: %s", exc)

    def _send_email(self, alert_type: str, message: str) -> None:
        """Send alert via SMTP email."""
        try:
            import smtplib
            from email.mime.text import MIMEText

            msg = MIMEText(message)
            msg["Subject"] = f"Trading Agent Alert: {alert_type}"
            msg["From"] = self.config.email_from
            msg["To"] = self.config.email_to

            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)
        except Exception as exc:
            logger.warning("Email alert error: %s", exc)
