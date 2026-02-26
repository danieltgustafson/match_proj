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

    # Broker selection: "paper", "alpaca", "ibkr", "schwab"
    broker: str = "paper"

    # Alpaca Markets -- https://alpaca.markets
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # Interactive Brokers -- https://interactivebrokers.com
    # Requires TWS or IB Gateway running locally
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497  # 7497=TWS paper, 7496=TWS live, 4002=GW paper, 4001=GW live
    ibkr_client_id: int = 1

    # Crypto exchange
    crypto_exchange: str = "coinbase"
    crypto_api_key: str = ""
    crypto_secret_key: str = ""

    # Data providers
    yahoo_finance_enabled: bool = True
    alpha_vantage_api_key: str = ""

    # News / Sentiment (optional)
    news_api_key: str = ""
    rapidapi_key: str = ""  # For Seeking Alpha via RapidAPI

    # Watchlist (comma-separated default symbols)
    watchlist: str = "AAPL,MSFT,GOOGL,AMZN,NVDA"

    # Trading schedule
    run_interval_seconds: int = 900  # 15 minutes

    # Risk management
    max_drawdown_pct: float = 0.15
    max_position_pct: float = 0.10
    max_daily_loss_pct: float = 0.03

    # Position sizing
    sizing_method: str = "kelly"  # kelly, risk_parity, fixed_fractional
    kelly_fraction: float = 0.5

    # Universe scanner (auto-discover symbols)
    scanner_enabled: bool = False
    scanner_universe: str = "all"  # all, sp500, growth, crypto, etfs
    scanner_max_symbols: int = 30
    scanner_rank_by: str = "composite"  # composite, momentum, volume, volatility
