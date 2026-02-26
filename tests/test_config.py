"""Tests for application configuration."""

from trading_agent.config import Settings


def test_default_settings() -> None:
    settings = Settings()
    assert settings.paper_trading is True
    assert settings.log_level == "INFO"
    assert settings.app_name == "trading-agent"
    assert settings.yahoo_finance_enabled is True
