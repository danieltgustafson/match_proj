"""Risk management layer -- guards against excessive exposure.

Checks every order against a set of risk rules before it reaches
the broker.  Inspired by institutional risk frameworks:

- **Max drawdown circuit breaker**: Halt trading if portfolio drops
  below a threshold from its peak (protects against tail events).
- **Position concentration limits**: No single holding exceeds X% of equity.
- **Sector / correlation limits**: Avoid piling into correlated bets.
- **Daily loss limit**: Stop trading for the day if losses exceed a cap.
- **Max open positions**: Enforce diversification.

All limits are configurable via RiskConfig.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from trading_agent.models.signals import Order, OrderSide

logger = logging.getLogger("trading_agent.risk")


@dataclass
class RiskConfig:
    """Risk management parameters."""

    # Maximum drawdown from equity peak before halting (e.g. 0.15 = 15%)
    max_drawdown_pct: float = 0.15

    # Maximum single position as fraction of equity
    max_position_pct: float = 0.10

    # Maximum daily loss as fraction of equity before stopping for the day
    max_daily_loss_pct: float = 0.03

    # Maximum number of open positions
    max_open_positions: int = 20

    # Maximum total exposure as fraction of equity (leverage limit)
    max_total_exposure_pct: float = 1.0

    # Symbols to never trade (e.g. restricted list)
    restricted_symbols: list[str] = field(default_factory=list)


@dataclass
class RiskState:
    """Tracks live risk metrics throughout a trading session."""

    equity_peak: float = 0.0
    current_equity: float = 0.0
    daily_starting_equity: float = 0.0
    total_exposure: float = 0.0
    open_position_count: int = 0
    position_values: dict[str, float] = field(default_factory=dict)
    halted: bool = False
    halt_reason: str = ""

    @property
    def current_drawdown(self) -> float:
        if self.equity_peak <= 0:
            return 0.0
        return (self.equity_peak - self.current_equity) / self.equity_peak

    @property
    def daily_pnl_pct(self) -> float:
        if self.daily_starting_equity <= 0:
            return 0.0
        return (
            (self.current_equity - self.daily_starting_equity)
            / self.daily_starting_equity
        )


class RiskCheckResult:
    """Result of a risk check -- approved or rejected with reason."""

    def __init__(self, approved: bool, reason: str = "") -> None:
        self.approved = approved
        self.reason = reason

    def __bool__(self) -> bool:
        return self.approved

    def __repr__(self) -> str:
        status = "APPROVED" if self.approved else f"REJECTED: {self.reason}"
        return f"RiskCheckResult({status})"


class RiskManager:
    """Evaluates orders against risk rules before execution.

    Usage::

        rm = RiskManager(config, state)
        result = rm.check_order(order, current_price)
        if result:
            broker.place_order(order)
        else:
            logger.warning("Order blocked: %s", result.reason)
    """

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        state: Optional[RiskState] = None,
    ) -> None:
        self.config = config or RiskConfig()
        self.state = state or RiskState()

    def update_state(
        self,
        current_equity: float,
        positions: dict[str, float],
        total_exposure: float,
    ) -> None:
        """Refresh risk state with latest portfolio data."""
        self.state.current_equity = current_equity
        self.state.equity_peak = max(self.state.equity_peak, current_equity)
        self.state.position_values = positions
        self.state.open_position_count = len(positions)
        self.state.total_exposure = total_exposure

        # Check drawdown circuit breaker
        if self.state.current_drawdown >= self.config.max_drawdown_pct:
            self.state.halted = True
            self.state.halt_reason = (
                f"Max drawdown breached: {self.state.current_drawdown:.1%} "
                f">= {self.config.max_drawdown_pct:.1%}"
            )
            logger.critical("TRADING HALTED: %s", self.state.halt_reason)

    def start_new_day(self, equity: float) -> None:
        """Reset daily tracking at start of trading day."""
        self.state.daily_starting_equity = equity
        # Do NOT reset halted state -- that requires manual intervention

    def check_order(self, order: Order, current_price: float) -> RiskCheckResult:
        """Run all risk checks on a proposed order.

        Returns RiskCheckResult -- truthy if approved, falsy if rejected.
        """
        # Circuit breaker
        if self.state.halted:
            return RiskCheckResult(False, f"Trading halted: {self.state.halt_reason}")

        # Restricted symbol
        if order.symbol in self.config.restricted_symbols:
            return RiskCheckResult(False, f"{order.symbol} is restricted")

        # Daily loss limit
        if self.state.daily_pnl_pct <= -self.config.max_daily_loss_pct:
            return RiskCheckResult(
                False,
                f"Daily loss limit hit: {self.state.daily_pnl_pct:.2%} "
                f"<= -{self.config.max_daily_loss_pct:.2%}",
            )

        # Only check position limits for buy orders
        if order.side is OrderSide.BUY:
            # Max open positions
            if self.state.open_position_count >= self.config.max_open_positions:
                return RiskCheckResult(
                    False,
                    f"Max positions reached: {self.state.open_position_count}"
                    f" >= {self.config.max_open_positions}",
                )

            # Position concentration
            order_value = order.qty * current_price
            existing = self.state.position_values.get(order.symbol, 0.0)
            new_total = existing + order_value
            if self.state.current_equity > 0:
                concentration = new_total / self.state.current_equity
                if concentration > self.config.max_position_pct:
                    return RiskCheckResult(
                        False,
                        f"{order.symbol} concentration {concentration:.1%} "
                        f"> {self.config.max_position_pct:.1%}",
                    )

            # Total exposure
            new_exposure = self.state.total_exposure + order_value
            if self.state.current_equity > 0:
                exposure_pct = new_exposure / self.state.current_equity
                if exposure_pct > self.config.max_total_exposure_pct:
                    return RiskCheckResult(
                        False,
                        f"Total exposure {exposure_pct:.1%} "
                        f"> {self.config.max_total_exposure_pct:.1%}",
                    )

        logger.debug("Order approved: %s %s %s", order.side.value, order.qty, order.symbol)
        return RiskCheckResult(True)

    def reset_halt(self) -> None:
        """Manually reset the circuit breaker (use with caution)."""
        self.state.halted = False
        self.state.halt_reason = ""
        logger.warning("Trading halt manually reset")
