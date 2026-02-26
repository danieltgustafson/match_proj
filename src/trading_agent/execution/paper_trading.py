"""Paper (simulated) trading broker for safe testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trading_agent.execution.broker import BaseBroker
from trading_agent.models.signals import Signal, SignalType


@dataclass
class PaperBroker(BaseBroker):
    """In-memory simulated broker.

    Tracks cash, positions, and an order log without touching real money.
    """

    initial_cash: float = 100_000.0
    cash: float = field(init=False)
    positions: dict[str, float] = field(default_factory=dict)
    order_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_cash

    def submit_order(self, signal: Signal) -> dict[str, Any]:
        order: dict[str, Any] = {
            "symbol": signal.symbol,
            "signal_type": signal.signal_type.value,
            "confidence": signal.confidence,
            "strategy": signal.strategy_name,
            "status": "filled" if signal.is_actionable else "skipped",
        }
        self.order_log.append(order)
        return order

    def get_positions(self) -> list[dict[str, Any]]:
        return [
            {"symbol": s, "qty": q, "market_value": 0.0}
            for s, q in self.positions.items()
        ]

    def get_balance(self) -> float:
        return self.cash

    def get_account(self) -> dict[str, Any]:
        return {
            "equity": self.cash,
            "cash": self.cash,
            "portfolio_value": self.cash,
            "buying_power": self.cash,
        }
