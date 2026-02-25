"""Abstract broker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from trading_agent.models.signals import Signal


class BaseBroker(ABC):
    """Interface for order execution backends."""

    @abstractmethod
    def submit_order(self, signal: Signal) -> dict[str, Any]:
        """Translate a Signal into a broker order and submit it.

        Returns a dict describing the order result.
        """

    @abstractmethod
    def get_positions(self) -> list[dict[str, Any]]:
        """Return current open positions."""

    @abstractmethod
    def get_balance(self) -> float:
        """Return current cash balance."""
