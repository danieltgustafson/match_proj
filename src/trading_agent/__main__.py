"""Entry point for running the trading agent as a module.

Usage:
    python -m trading_agent                 # single pass (paper trading)
    python -m trading_agent --live          # single pass (live trading)
    python -m trading_agent --schedule      # run on interval
    python -m trading_agent --schedule --interval 300  # every 5 min
"""

from __future__ import annotations

import argparse
import sys

from trading_agent.config import Settings
from trading_agent.runner import TradingRunner
from trading_agent.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Agent")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live trading (overrides PAPER_TRADING setting)",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run on a recurring schedule instead of a single pass",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Seconds between scheduled runs (default: from config)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Maximum number of scheduled iterations (default: unlimited)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols to trade",
    )
    args = parser.parse_args()

    settings = Settings()
    if args.live:
        settings.paper_trading = False

    setup_logging(settings.log_level)

    watchlist = (
        args.symbols.split(",") if args.symbols else settings.watchlist.split(",")
    )

    runner = TradingRunner.from_settings(
        settings=settings,
        watchlist=watchlist,
    )

    mode = "PAPER" if settings.paper_trading else "LIVE"
    print(f"Trading Agent v0.1.0 [{mode} mode]")
    print(f"Watchlist: {watchlist}")
    print(f"Sizing: {settings.sizing_method} (kelly_fraction={settings.kelly_fraction})")
    print(f"Risk: max_drawdown={settings.max_drawdown_pct:.0%}, "
          f"max_position={settings.max_position_pct:.0%}")
    print()

    if args.schedule:
        interval = args.interval or settings.run_interval_seconds
        print(f"Running on schedule (every {interval}s)...")
        runner.run_scheduled(
            interval_seconds=interval,
            max_iterations=args.max_runs,
        )
    else:
        result = runner.run_once()
        print(result.summary())
        for trade in result.trades:
            print(
                f"  {trade.side.upper()} {trade.symbol} "
                f"x{trade.qty:.0f} @ ${trade.price:.2f} "
                f"[{trade.strategy}]"
            )
        if result.errors:
            print("Errors:")
            for err in result.errors:
                print(f"  - {err}")


if __name__ == "__main__":
    main()
