"""Application configuration via environment variables and .env files."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration loaded from environment / .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # General
    app_name: str = "trading-agent"
    log_level: str = "INFO"
    paper_trading: bool = True

    # Alpaca (stock broker)
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # Crypto exchange
    crypto_exchange: str = "coinbase"
    crypto_api_key: str = ""
    crypto_secret_key: str = ""

    # Data providers
    yahoo_finance_enabled: bool = True
