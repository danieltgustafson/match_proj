"""Trading runner -- the main orchestrator that actually executes trades.

This is the top-level component that ties everything together:

1. Fetches market data from configured providers
2. Runs each symbol through the strategy ensemble
3. Passes signals to the portfolio manager for sizing
4. Sends sized orders through the risk manager for approval
5. Submits approved orders to the broker
6. Logs everything and tracks performance

Usage::

    from trading_agent.runner import TradingRunner

    runner = TradingRunner.from_settings()
    runner.run_once()           # single pass through the watchlist
    runner.run_scheduled()      # run on a schedule (e.g. every 15 min)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from trading_agent.config import Settings
from trading_agent.data.providers.alpha_vantage import AlphaVantageProvider
from trading_agent.execution.alpaca_broker import AlpacaBroker
from trading_agent.execution.paper_trading import PaperBroker
from trading_agent.execution.portfolio_manager import PortfolioConfig, PortfolioManager
from trading_agent.execution.risk_manager import RiskConfig, RiskManager, RiskState
from trading_agent.models.signals import (
    Order,
    OrderSide,
    Signal,
    SignalType,
)
from trading_agent.strategies.base import BaseStrategy
from trading_agent.strategies.ensemble import EnsembleStrategy
from trading_agent.strategies.macd_crossover import MACDCrossoverStrategy
from trading_agent.strategies.mean_reversion import MeanReversionStrategy
from trading_agent.strategies.momentum import MomentumStrategy

logger = logging.getLogger("trading_agent.runner")


@dataclass
class TradeRecord:
    """Record of a single trade execution."""

    timestamp: str
    symbol: str
    side: str
    qty: float
    price: float
    strategy: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of a single run pass."""

    timestamp: str
    symbols_evaluated: int = 0
    signals_generated: int = 0
    orders_submitted: int = 0
    orders_rejected: int = 0
    trades: list[TradeRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.timestamp}] Evaluated {self.symbols_evaluated} symbols | "
            f"Signals: {self.signals_generated} | "
            f"Orders: {self.orders_submitted} submitted, "
            f"{self.orders_rejected} rejected | "
            f"Errors: {len(self.errors)}"
        )


class TradingRunner:
    """Main trading loop that orchestrates data -> strategy -> execution.

    Parameters
    ----------
    broker:
        The broker to execute orders through (Alpaca or Paper).
    data_provider:
        Market data source.
    strategy:
        The strategy (typically an EnsembleStrategy) to generate signals.
    watchlist:
        List of ticker symbols to evaluate each cycle.
    portfolio_config:
        Position sizing parameters.
    risk_config:
        Risk management parameters.
    """

    def __init__(
        self,
        broker: Any,
        data_provider: Any,
        strategy: BaseStrategy,
        watchlist: list[str],
        portfolio_config: Optional[PortfolioConfig] = None,
        risk_config: Optional[RiskConfig] = None,
    ) -> None:
        self.broker = broker
        self.data_provider = data_provider
        self.strategy = strategy
        self.watchlist = watchlist
        self.portfolio_config = portfolio_config or PortfolioConfig()
        self.risk_config = risk_config or RiskConfig()
        self.risk_manager = RiskManager(self.risk_config)
        self.trade_history: list[TradeRecord] = []

    @classmethod
    def from_settings(
        cls,
        settings: Optional[Settings] = None,
        watchlist: Optional[list[str]] = None,
        strategy: Optional[BaseStrategy] = None,
    ) -> TradingRunner:
        """Factory that builds a fully wired runner from Settings.

        This is the recommended way to create a runner -- it reads your
        .env config and wires up all components automatically.
        """
        settings = settings or Settings()

        # Broker
        if settings.paper_trading:
            broker = PaperBroker(initial_cash=100_000.0)
            logger.info("Using PAPER trading broker")
        else:
            broker = AlpacaBroker(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                base_url=settings.alpaca_base_url,
            )
            logger.info("Using LIVE Alpaca broker at %s", settings.alpaca_base_url)

        # Data provider
        data_provider = AlphaVantageProvider(api_key=settings.alpha_vantage_api_key)

        # Default ensemble strategy if none provided
        if strategy is None:
            strategy = EnsembleStrategy(
                strategies={
                    "momentum": MomentumStrategy(rsi_period=14),
                    "mean_reversion": MeanReversionStrategy(window=20),
                    "macd": MACDCrossoverStrategy(),
                },
                min_agreement=0.5,
                weighting_mode="confidence",
            )

        # Default watchlist
        if watchlist is None:
            watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

        return cls(
            broker=broker,
            data_provider=data_provider,
            strategy=strategy,
            watchlist=watchlist,
        )

    def run_once(self) -> RunResult:
        """Execute a single pass: evaluate all symbols and submit orders.

        This is the core trading loop. Call it on a schedule or manually.
        """
        result = RunResult(timestamp=datetime.utcnow().isoformat())

        # Refresh portfolio state
        equity, cash, positions = self._get_portfolio_state()

        # Update risk manager
        position_values = {
            p.get("symbol", ""): p.get("market_value", 0.0)
            for p in (positions if isinstance(positions, list) else [])
        }
        total_exposure = sum(abs(v) for v in position_values.values())
        self.risk_manager.update_state(equity, position_values, total_exposure)

        if self.risk_manager.state.halted:
            result.errors.append(f"Trading halted: {self.risk_manager.state.halt_reason}")
            logger.critical("Run aborted -- trading is halted")
            return result

        # Build current positions map for portfolio manager
        held_positions = {
            p.get("symbol", ""): p.get("qty", 0.0)
            for p in (positions if isinstance(positions, list) else [])
        }

        pm = PortfolioManager(
            equity=equity,
            cash=cash,
            config=self.portfolio_config,
            current_positions=held_positions,
        )

        for symbol in self.watchlist:
            result.symbols_evaluated += 1
            try:
                # Fetch data
                data = self.data_provider.get_historical(symbol)
                if data.empty:
                    logger.warning("No data for %s, skipping", symbol)
                    continue

                # Generate signal
                signal = self.strategy.evaluate(data)
                signal = Signal(
                    signal_type=signal.signal_type,
                    symbol=symbol,
                    confidence=signal.confidence,
                    strategy_name=signal.strategy_name,
                    metadata=signal.metadata,
                )
                result.signals_generated += 1

                if not signal.is_actionable:
                    continue

                # Get current price for sizing
                current_price = float(data["close"].iloc[-1])

                # Size the order
                order = pm.signal_to_order(
                    signal,
                    current_price=current_price,
                    volatility=float(data["close"].pct_change().std())
                    if len(data) > 1
                    else 0.02,
                )
                if order is None:
                    continue

                # Risk check
                risk_result = self.risk_manager.check_order(order, current_price)
                if not risk_result:
                    result.orders_rejected += 1
                    logger.info(
                        "Order rejected for %s: %s", symbol, risk_result.reason
                    )
                    continue

                # Execute
                exec_result = self.broker.submit_order(signal)
                result.orders_submitted += 1

                trade = TradeRecord(
                    timestamp=datetime.utcnow().isoformat(),
                    symbol=symbol,
                    side=signal.signal_type.value,
                    qty=order.qty,
                    price=current_price,
                    strategy=signal.strategy_name,
                    status=exec_result.get("status", "unknown"),
                    metadata=signal.metadata,
                )
                result.trades.append(trade)
                self.trade_history.append(trade)

                logger.info(
                    "TRADE: %s %s %.0f shares @ $%.2f (strategy=%s, confidence=%.2f)",
                    signal.signal_type.value.upper(),
                    symbol,
                    order.qty,
                    current_price,
                    signal.strategy_name,
                    signal.confidence,
                )

            except Exception as exc:
                error_msg = f"Error processing {symbol}: {exc}"
                result.errors.append(error_msg)
                logger.error(error_msg, exc_info=True)

        logger.info(result.summary())
        return result

    def run_scheduled(
        self,
        interval_seconds: int = 900,
        max_iterations: Optional[int] = None,
    ) -> list[RunResult]:
        """Run the trading loop on a fixed interval.

        Parameters
        ----------
        interval_seconds:
            Seconds between each run (default 15 minutes).
        max_iterations:
            Stop after this many iterations (None = run forever).
        """
        results = []
        iteration = 0

        logger.info(
            "Starting scheduled trading loop (interval=%ds, watchlist=%s)",
            interval_seconds,
            self.watchlist,
        )

        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            logger.info("=== Run iteration %d ===", iteration)

            result = self.run_once()
            results.append(result)

            if max_iterations is not None and iteration >= max_iterations:
                break

            logger.info("Sleeping %d seconds...", interval_seconds)
            time.sleep(interval_seconds)

        return results

    def _get_portfolio_state(self) -> tuple[float, float, list[dict[str, Any]]]:
        """Get current equity, cash, and positions from broker."""
        try:
            if hasattr(self.broker, "get_account"):
                account = self.broker.get_account()
                equity = account.get("equity", account.get("portfolio_value", 0.0))
                cash = account.get("cash", 0.0)
            else:
                equity = self.broker.get_balance()
                cash = equity
            positions = self.broker.get_positions()
            return equity, cash, positions
        except Exception as exc:
            logger.error("Failed to get portfolio state: %s", exc)
            return 100_000.0, 100_000.0, []
