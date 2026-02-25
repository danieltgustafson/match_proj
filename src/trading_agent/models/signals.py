"""Trading signal models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SignalType(Enum):
    """Direction of a trading signal."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True)
class Signal:
    """Immutable trading signal emitted by a strategy.

    Attributes
    ----------
    signal_type:
        BUY, SELL, or HOLD.
    confidence:
        A value in [0, 1] indicating how strong the signal is.
    metadata:
        Arbitrary extra info (indicator values, reasoning, etc.).
    """

    signal_type: SignalType
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Return True if the signal is not HOLD."""
        return self.signal_type is not SignalType.HOLD
