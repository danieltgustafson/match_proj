"""Entry point for running the trading agent as a module."""

from trading_agent.config import Settings


def main() -> None:
    settings = Settings()
    print(f"Trading Agent v{settings.app_name}")
    print(f"Paper trading: {settings.paper_trading}")
    print("Ready. Configure a strategy and agent to begin.")


if __name__ == "__main__":
    main()
