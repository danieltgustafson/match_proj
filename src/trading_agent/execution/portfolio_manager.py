"""Portfolio manager -- translates signals into sized orders.

Implements multiple position-sizing methods drawn from quantitative
finance research:

- **Kelly Criterion**: Optimal fraction of capital to risk per trade,
  based on estimated win probability and payoff ratio.  Originally from
  J.L. Kelly (1956), widely used in quant finance (Ed Thorp, etc.).

- **Risk Parity**: Size positions so each contributes equal risk
  (measured by volatility) to the portfolio.

- **Fixed Fractional**: Simple approach -- risk a fixed % of equity per trade.

The portfolio manager also enforces per-position limits and coordinates
with the RiskManager before allowing orders through.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from trading_agent.models.signals import (
    Order,
    OrderSide,
    OrderType,
    Signal,
    SignalType,
    TimeInForce,
)

logger = logging.getLogger("trading_agent.portfolio")


@dataclass
class PortfolioConfig:
    """Tunable portfolio parameters."""

    # Maximum fraction of equity in any single position
    max_position_pct: float = 0.10

    # Default sizing method: "kelly", "risk_parity", "fixed_fractional"
    sizing_method: str = "kelly"

    # For fixed fractional: risk this % of equity per trade
    fixed_fraction: float = 0.02

    # Kelly fraction cap (full Kelly is aggressive; half-Kelly is common)
    kelly_fraction: float = 0.5

    # Minimum order size in dollars (skip tiny orders)
    min_order_value: float = 100.0

    # Maximum number of open positions
    max_positions: int = 20


class PortfolioManager:
    """Translate signals into properly sized orders.

    This sits between the strategy layer (which produces Signals) and
    the broker layer (which executes Orders).  It decides *how much*
    to buy or sell based on the portfolio state, risk budget, and the
    signal's confidence.

    Parameters
    ----------
    equity:
        Current total portfolio equity.
    cash:
        Available cash for new positions.
    config:
        Portfolio sizing parameters.
    current_positions:
        Dict mapping symbol -> current qty held.
    """

    def __init__(
        self,
        equity: float,
        cash: float,
        config: Optional[PortfolioConfig] = None,
        current_positions: Optional[dict[str, float]] = None,
    ) -> None:
        self.equity = equity
        self.cash = cash
        self.config = config or PortfolioConfig()
        self.current_positions = current_positions or {}

    def signal_to_order(
        self,
        signal: Signal,
        current_price: float,
        win_rate: float = 0.55,
        avg_win: float = 1.5,
        avg_loss: float = 1.0,
        volatility: float = 0.02,
    ) -> Optional[Order]:
        """Convert a Signal into a sized Order, or None if no trade.

        Parameters
        ----------
        signal:
            The trading signal to act on.
        current_price:
            Current market price for the symbol.
        win_rate:
            Estimated probability of a winning trade (for Kelly).
        avg_win:
            Average win size relative to risk (reward/risk ratio).
        avg_loss:
            Average loss size relative to risk.
        volatility:
            Recent daily return volatility (for risk parity sizing).
        """
        if not signal.is_actionable or current_price <= 0:
            return None

        side = OrderSide.BUY if signal.signal_type is SignalType.BUY else OrderSide.SELL

        # For sell signals, check if we actually hold the position
        if side is OrderSide.SELL:
            held = self.current_positions.get(signal.symbol, 0)
            if held <= 0:
                logger.info(
                    "Skipping SELL for %s -- no position held", signal.symbol
                )
                return None
            # Sell entire position (or partial based on confidence)
            sell_qty = held * min(signal.confidence + 0.5, 1.0)
            sell_qty = max(1.0, math.floor(sell_qty))
            return Order(
                symbol=signal.symbol,
                side=OrderSide.SELL,
                qty=sell_qty,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                metadata={"strategy": signal.strategy_name},
            )

        # Buy sizing
        method = self.config.sizing_method
        if method == "kelly":
            fraction = self._kelly_size(
                win_rate, avg_win, avg_loss, signal.confidence
            )
        elif method == "risk_parity":
            fraction = self._risk_parity_size(volatility, signal.confidence)
        else:
            fraction = self._fixed_fractional_size(signal.confidence)

        # Cap at max position size
        fraction = min(fraction, self.config.max_position_pct)

        dollar_amount = self.equity * fraction
        if dollar_amount < self.config.min_order_value:
            logger.info(
                "Order for %s too small ($%.2f < $%.2f minimum), skipping",
                signal.symbol,
                dollar_amount,
                self.config.min_order_value,
            )
            return None

        if dollar_amount > self.cash:
            dollar_amount = self.cash
            if dollar_amount < self.config.min_order_value:
                logger.info("Insufficient cash for %s, skipping", signal.symbol)
                return None

        qty = math.floor(dollar_amount / current_price)
        if qty < 1:
            return None

        logger.info(
            "Sizing %s: method=%s fraction=%.4f -> %d shares ($%.2f)",
            signal.symbol,
            method,
            fraction,
            qty,
            qty * current_price,
        )

        return Order(
            symbol=signal.symbol,
            side=OrderSide.BUY,
            qty=float(qty),
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            metadata={"strategy": signal.strategy_name, "sizing_method": method},
        )

    # ------------------------------------------------------------------
    # Sizing methods
    # ------------------------------------------------------------------

    def _kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        confidence: float,
    ) -> float:
        """Kelly Criterion position sizing.

        Kelly fraction = (p * b - q) / b
        where p = win probability, q = 1-p, b = win/loss ratio.

        We apply a fractional Kelly (default half-Kelly) to reduce
        variance, then scale by signal confidence.

        References:
        - Kelly, J.L. (1956) "A New Interpretation of Information Rate"
        - Thorp, E.O. (2006) "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
        """
        if avg_loss == 0:
            return 0.0

        b = avg_win / avg_loss
        p = win_rate
        q = 1.0 - p

        kelly = (p * b - q) / b
        kelly = max(kelly, 0.0)

        # Apply fractional Kelly and confidence scaling
        sized = kelly * self.config.kelly_fraction * confidence
        return sized

    def _risk_parity_size(self, volatility: float, confidence: float) -> float:
        """Risk parity sizing -- allocate inversely proportional to volatility.

        Target: each position contributes ~equal risk.
        Position size = target_risk / volatility

        Based on Bridgewater's All Weather approach and
        Maillard, Roncalli & Teiletche (2010).
        """
        if volatility <= 0:
            return 0.0

        # Target 1% daily portfolio risk per position
        target_risk = 0.01
        raw = target_risk / volatility

        # Scale by confidence, cap at max position
        sized = raw * confidence
        return min(sized, self.config.max_position_pct)

    def _fixed_fractional_size(self, confidence: float) -> float:
        """Fixed fractional sizing -- risk a fixed % of equity.

        Simplest approach. The confidence score scales within the
        fixed fraction.
        """
        return self.config.fixed_fraction * confidence
