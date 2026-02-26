"""Trading signal and order models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class SignalType(Enum):
    """Direction of a trading signal."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"          # good til cancelled
    IOC = "ioc"          # immediate or cancel
    FOK = "fok"          # fill or kill
    OPG = "opg"          # market on open
    CLS = "cls"          # market on close


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Signal:
    """Immutable trading signal emitted by a strategy.

    Attributes
    ----------
    signal_type:
        BUY, SELL, or HOLD.
    symbol:
        Ticker symbol this signal applies to (e.g. "AAPL", "BTC/USD").
    confidence:
        A value in [0, 1] indicating how strong the signal is.
    strategy_name:
        Name of the strategy that generated this signal.
    metadata:
        Arbitrary extra info (indicator values, reasoning, etc.).
    """

    signal_type: SignalType
    symbol: str = ""
    confidence: float = 0.0
    strategy_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Return True if the signal is not HOLD."""
        return self.signal_type is not SignalType.HOLD


@dataclass
class Order:
    """Represents a trade order to be submitted to a broker.

    Created by the portfolio manager after translating a Signal into
    a concrete position size and order type.
    """

    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    broker_order_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_qty: float = 0.0
    created_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "qty": self.qty,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "broker_order_id": self.broker_order_id,
            "filled_price": self.filled_price,
            "filled_qty": self.filled_qty,
        }


@dataclass
class Position:
    """Represents a current holding."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    side: str = "long"

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def is_short(self) -> bool:
        return self.qty < 0
