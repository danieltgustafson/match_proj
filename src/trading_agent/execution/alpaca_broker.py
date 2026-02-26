"""Alpaca Markets broker integration for live and paper trading.

Alpaca provides commission-free stock and crypto trading with a REST API.
Docs: https://docs.alpaca.markets/

This module wraps the Alpaca API to implement our BaseBroker interface,
enabling real order execution in both paper and live modes.

You need:
- ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env
- For paper trading: ALPACA_BASE_URL=https://paper-api.alpaca.markets
- For live trading: ALPACA_BASE_URL=https://api.alpaca.markets
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import httpx

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

logger = logging.getLogger("trading_agent.alpaca")


class AlpacaError(Exception):
    """Raised when the Alpaca API returns an error."""


class AlpacaBroker(BaseBroker):
    """Broker that submits real orders to Alpaca Markets.

    Parameters
    ----------
    api_key:
        Alpaca API key ID.
    secret_key:
        Alpaca API secret key.
    base_url:
        API base URL. Use paper-api.alpaca.markets for paper trading.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
    ) -> None:
        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca API key and secret key are required. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file."
            )
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # BaseBroker interface
    # ------------------------------------------------------------------

    def submit_order(self, signal: Signal) -> dict[str, Any]:
        """Translate a Signal into an Alpaca order and submit it.

        This is a convenience method -- for more control, use
        `place_order()` with a fully specified Order object.
        """
        if not signal.is_actionable:
            return {"status": "skipped", "reason": "HOLD signal"}

        side = OrderSide.BUY if signal.signal_type is SignalType.BUY else OrderSide.SELL
        # Default to 1 share -- real sizing is done by the PortfolioManager
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
        """Return current open positions from Alpaca."""
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
    # Alpaca-specific methods
    # ------------------------------------------------------------------

    def get_account(self) -> dict[str, Any]:
        """Fetch account details (equity, cash, buying power, etc.)."""
        data = self._get("/v2/account")
        return {
            "id": data.get("id"),
            "status": data.get("status"),
            "cash": float(data.get("cash", 0)),
            "portfolio_value": float(data.get("portfolio_value", 0)),
            "equity": float(data.get("equity", 0)),
            "buying_power": float(data.get("buying_power", 0)),
            "long_market_value": float(data.get("long_market_value", 0)),
            "short_market_value": float(data.get("short_market_value", 0)),
            "pattern_day_trader": data.get("pattern_day_trader", False),
            "trading_blocked": data.get("trading_blocked", False),
        }

    def place_order(self, order: Order) -> dict[str, Any]:
        """Submit a fully specified Order to Alpaca.

        Returns a dict with order details including broker_order_id.
        """
        payload: dict[str, Any] = {
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": order.side.value,
            "type": order.order_type.value,
            "time_in_force": order.time_in_force.value,
        }
        if order.limit_price is not None:
            payload["limit_price"] = str(order.limit_price)
        if order.stop_price is not None:
            payload["stop_price"] = str(order.stop_price)

        logger.info(
            "Submitting order: %s %s %s @ %s",
            order.side.value,
            order.qty,
            order.symbol,
            order.order_type.value,
        )

        data = self._post("/v2/orders", json_body=payload)

        order.broker_order_id = data.get("id")
        order.status = self._map_status(data.get("status", ""))
        if data.get("filled_avg_price"):
            order.filled_price = float(data["filled_avg_price"])
        if data.get("filled_qty"):
            order.filled_qty = float(data["filled_qty"])

        logger.info(
            "Order %s status=%s filled_qty=%s",
            order.broker_order_id,
            order.status.value,
            order.filled_qty,
        )
        return order.to_dict()

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an open order by its Alpaca order ID."""
        self._delete(f"/v2/orders/{order_id}")
        return {"order_id": order_id, "status": "cancelled"}

    def list_orders(
        self,
        status: str = "open",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List orders with optional status filter."""
        data = self._get("/v2/orders", params={"status": status, "limit": str(limit)})
        return data if isinstance(data, list) else []

    def list_positions(self) -> list[Position]:
        """Return typed Position objects for all current holdings."""
        data = self._get("/v2/positions")
        positions = []
        for item in (data if isinstance(data, list) else []):
            positions.append(
                Position(
                    symbol=item["symbol"],
                    qty=float(item.get("qty", 0)),
                    avg_entry_price=float(item.get("avg_entry_price", 0)),
                    current_price=float(item.get("current_price", 0)),
                    market_value=float(item.get("market_value", 0)),
                    unrealized_pnl=float(item.get("unrealized_pnl", 0)),
                    side=item.get("side", "long"),
                )
            )
        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol, or None if not held."""
        try:
            item = self._get(f"/v2/positions/{symbol}")
            return Position(
                symbol=item["symbol"],
                qty=float(item.get("qty", 0)),
                avg_entry_price=float(item.get("avg_entry_price", 0)),
                current_price=float(item.get("current_price", 0)),
                market_value=float(item.get("market_value", 0)),
                unrealized_pnl=float(item.get("unrealized_pnl", 0)),
                side=item.get("side", "long"),
            )
        except AlpacaError:
            return None

    def is_market_open(self) -> bool:
        """Check if the US stock market is currently open."""
        data = self._get("/v2/clock")
        return bool(data.get("is_open", False))

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict[str, str]] = None) -> Any:
        resp = httpx.get(
            f"{self.base_url}{path}",
            headers=self._headers,
            params=params,
            timeout=30.0,
        )
        return self._handle_response(resp)

    def _post(self, path: str, json_body: dict[str, Any]) -> Any:
        resp = httpx.post(
            f"{self.base_url}{path}",
            headers=self._headers,
            json=json_body,
            timeout=30.0,
        )
        return self._handle_response(resp)

    def _delete(self, path: str) -> Any:
        resp = httpx.delete(
            f"{self.base_url}{path}",
            headers=self._headers,
            timeout=30.0,
        )
        if resp.status_code == 204:
            return {}
        return self._handle_response(resp)

    @staticmethod
    def _handle_response(resp: httpx.Response) -> Any:
        if resp.status_code >= 400:
            try:
                body = resp.json()
                message = body.get("message", resp.text)
            except Exception:
                message = resp.text
            raise AlpacaError(f"Alpaca API error ({resp.status_code}): {message}")
        return resp.json()

    @staticmethod
    def _map_status(raw: str) -> OrderStatus:
        mapping = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.SUBMITTED,
            "pending_new": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
        }
        return mapping.get(raw.lower(), OrderStatus.PENDING)
