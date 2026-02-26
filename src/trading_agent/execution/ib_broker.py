"""Interactive Brokers integration via the ib_insync library.

IBKR is the gold standard for retail algorithmic trading -- it supports
stocks, options, futures, forex, and crypto with a robust API.

Prerequisites:
1. Install TWS (Trader Workstation) or IB Gateway
2. Enable API connections in TWS: Configure > API > Settings
   - Check "Enable ActiveX and Socket Clients"
   - Set socket port (default: 7497 for paper, 7496 for live)
   - Add 127.0.0.1 to trusted IPs
3. pip install ib_insync

Usage::

    from trading_agent.execution.ib_broker import IBBroker

    broker = IBBroker(host="127.0.0.1", port=7497)  # paper trading
    broker.connect()
    account = broker.get_account()
    positions = broker.list_positions()
    broker.disconnect()

Docs: https://ib-insync.readthedocs.io/
IB API: https://interactivebrokers.github.io/
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from trading_agent.execution.broker import BaseBroker
from trading_agent.models.signals import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Signal,
    SignalType,
    TimeInForce,
)

logger = logging.getLogger("trading_agent.ib")


class IBConnectionError(Exception):
    """Raised when unable to connect to TWS/IB Gateway."""


class IBBroker(BaseBroker):
    """Broker that executes orders through Interactive Brokers.

    Requires TWS or IB Gateway to be running locally (or on a
    reachable host).

    Parameters
    ----------
    host:
        TWS/Gateway host (default: 127.0.0.1).
    port:
        TWS/Gateway port.
        - 7497 = TWS paper trading
        - 7496 = TWS live trading
        - 4002 = IB Gateway paper
        - 4001 = IB Gateway live
    client_id:
        Unique client ID for this connection (use different IDs
        for multiple concurrent connections).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib: Any = None  # ib_insync.IB instance
        self._connected = False

    def connect(self) -> None:
        """Connect to TWS / IB Gateway."""
        try:
            from ib_insync import IB
        except ImportError as exc:
            raise ImportError(
                "ib_insync is required for IBBroker. "
                "Install it with: pip install ib_insync"
            ) from exc

        self._ib = IB()
        try:
            self._ib.connect(self.host, self.port, clientId=self.client_id)
            self._connected = True
            logger.info(
                "Connected to IB at %s:%d (client_id=%d)",
                self.host,
                self.port,
                self.client_id,
            )
        except Exception as exc:
            raise IBConnectionError(
                f"Failed to connect to TWS/IB Gateway at {self.host}:{self.port}. "
                f"Make sure TWS or IB Gateway is running with API connections enabled. "
                f"Error: {exc}"
            ) from exc

    def disconnect(self) -> None:
        """Disconnect from TWS / IB Gateway."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")

    def _ensure_connected(self) -> None:
        if not self._connected or self._ib is None:
            raise IBConnectionError("Not connected to IB. Call connect() first.")

    # ------------------------------------------------------------------
    # BaseBroker interface
    # ------------------------------------------------------------------

    def submit_order(self, signal: Signal) -> dict[str, Any]:
        """Translate a Signal into an IB order and submit it."""
        if not signal.is_actionable:
            return {"status": "skipped", "reason": "HOLD signal"}

        side = OrderSide.BUY if signal.signal_type is SignalType.BUY else OrderSide.SELL
        qty = signal.metadata.get("qty", 1)

        order = Order(
            symbol=signal.symbol,
            side=side,
            qty=qty,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
        )
        return self.place_order(order)

    def get_positions(self) -> list[dict[str, Any]]:
        """Return current open positions."""
        positions = self.list_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry_price": p.avg_entry_price,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
            }
            for p in positions
        ]

    def get_balance(self) -> float:
        """Return current cash balance."""
        account = self.get_account()
        return account.get("cash", 0.0)

    # ------------------------------------------------------------------
    # IB-specific methods
    # ------------------------------------------------------------------

    def get_account(self) -> dict[str, Any]:
        """Fetch account summary from IB."""
        self._ensure_connected()
        from ib_insync import IB

        account_values = self._ib.accountSummary()
        result: dict[str, Any] = {}
        for av in account_values:
            key = av.tag
            try:
                val: Any = float(av.value)
            except (ValueError, TypeError):
                val = av.value

            if key == "NetLiquidation":
                result["equity"] = val
                result["portfolio_value"] = val
            elif key == "TotalCashValue":
                result["cash"] = val
            elif key == "BuyingPower":
                result["buying_power"] = val
            elif key == "GrossPositionValue":
                result["total_exposure"] = val

        return result

    def place_order(self, order: Order) -> dict[str, Any]:
        """Submit a fully specified Order to Interactive Brokers."""
        self._ensure_connected()
        from ib_insync import LimitOrder, MarketOrder, StopOrder, Stock

        # Create IB contract
        contract = Stock(order.symbol, "SMART", "USD")

        # Create IB order
        action = "BUY" if order.side is OrderSide.BUY else "SELL"

        if order.order_type is OrderType.MARKET:
            ib_order = MarketOrder(action, order.qty)
        elif order.order_type is OrderType.LIMIT and order.limit_price is not None:
            ib_order = LimitOrder(action, order.qty, order.limit_price)
        elif order.order_type is OrderType.STOP and order.stop_price is not None:
            ib_order = StopOrder(action, order.qty, order.stop_price)
        else:
            ib_order = MarketOrder(action, order.qty)

        logger.info(
            "Submitting IB order: %s %s %s @ %s",
            action,
            order.qty,
            order.symbol,
            order.order_type.value,
        )

        trade = self._ib.placeOrder(contract, ib_order)

        # Wait briefly for order acknowledgement
        self._ib.sleep(1)

        order.broker_order_id = str(trade.order.orderId)
        order.status = self._map_ib_status(trade.orderStatus.status)

        if trade.orderStatus.avgFillPrice:
            order.filled_price = trade.orderStatus.avgFillPrice
        order.filled_qty = trade.orderStatus.filled

        logger.info(
            "IB order %s status=%s filled=%s",
            order.broker_order_id,
            order.status.value,
            order.filled_qty,
        )
        return order.to_dict()

    def cancel_order(self, trade: Any) -> None:
        """Cancel an open IB order."""
        self._ensure_connected()
        self._ib.cancelOrder(trade.order)

    def list_positions(self) -> list[Position]:
        """Return typed Position objects for all current IB holdings."""
        self._ensure_connected()
        ib_positions = self._ib.positions()
        positions = []
        for pos in ib_positions:
            positions.append(
                Position(
                    symbol=pos.contract.symbol,
                    qty=pos.position,
                    avg_entry_price=pos.avgCost,
                    current_price=0.0,  # would need market data subscription
                    market_value=pos.position * pos.avgCost,
                    unrealized_pnl=0.0,
                    side="long" if pos.position > 0 else "short",
                )
            )
        return positions

    def get_market_price(self, symbol: str) -> Optional[float]:
        """Get the last traded price for a symbol (requires market data)."""
        self._ensure_connected()
        from ib_insync import Stock

        contract = Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)
        ticker = self._ib.reqMktData(contract)
        self._ib.sleep(2)  # wait for data
        price = ticker.last if ticker.last > 0 else ticker.close
        self._ib.cancelMktData(contract)
        return float(price) if price > 0 else None

    def is_market_open(self) -> bool:
        """Check if the exchange is currently in regular trading hours."""
        self._ensure_connected()
        # IB doesn't have a direct "is market open" call;
        # we check via the exchange calendar
        import datetime

        now = datetime.datetime.now()
        # Simple US market hours check (9:30 AM - 4:00 PM ET)
        # For production, use IB's trading hours from contract details
        weekday = now.weekday()
        if weekday >= 5:  # Weekend
            return False
        return True  # Simplified; real impl would check ET hours

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_ib_status(raw: str) -> OrderStatus:
        mapping = {
            "Submitted": OrderStatus.SUBMITTED,
            "PreSubmitted": OrderStatus.PENDING,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Inactive": OrderStatus.REJECTED,
            "PendingSubmit": OrderStatus.PENDING,
            "PendingCancel": OrderStatus.PENDING,
            "ApiCancelled": OrderStatus.CANCELLED,
        }
        return mapping.get(raw, OrderStatus.PENDING)
