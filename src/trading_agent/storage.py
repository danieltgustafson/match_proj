"""Trade log persistence via SQLite.

Stores all trades, signals, portfolio snapshots, and scanner results
so you can review history after restarts. SQLite requires no server
setup -- just a file on disk.

Tables:
- trades: every executed trade with timestamp, symbol, side, qty, price
- signals: every signal generated (including HOLDs for analysis)
- portfolio_snapshots: periodic equity/cash/position snapshots
- scanner_results: what the scanner found each cycle
- alerts: triggered alerts for review

Usage::

    from trading_agent.storage import TradeStore

    store = TradeStore("trading_history.db")
    store.log_trade(trade_record)
    recent = store.get_trades(limit=50)
    pnl = store.get_daily_pnl()
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("trading_agent.storage")


class TradeStore:
    """SQLite-backed storage for trade history and portfolio state."""

    def __init__(self, db_path: str = "trading_history.db") -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_tables()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_tables(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                strategy TEXT,
                status TEXT,
                confidence REAL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                confidence REAL,
                strategy TEXT,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                cash REAL NOT NULL,
                num_positions INTEGER,
                positions TEXT,
                total_exposure REAL
            );

            CREATE TABLE IF NOT EXISTS scanner_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                num_scanned INTEGER,
                num_passed INTEGER,
                top_symbols TEXT
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                acknowledged INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
            CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp ON portfolio_snapshots(timestamp);
        """)
        conn.commit()
        logger.info("Trade store initialized at %s", self.db_path)

    # ------------------------------------------------------------------
    # Trade logging
    # ------------------------------------------------------------------

    def log_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        strategy: str = "",
        status: str = "",
        confidence: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Record a trade execution."""
        ts = timestamp or datetime.utcnow().isoformat()
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO trades (timestamp, symbol, side, qty, price,
               strategy, status, confidence, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, symbol, side, qty, price, strategy, status, confidence,
             json.dumps(metadata or {})),
        )
        conn.commit()
        return cursor.lastrowid or 0

    def log_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float = 0.0,
        strategy: str = "",
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        """Record a signal (including HOLDs for analysis)."""
        ts = timestamp or datetime.utcnow().isoformat()
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO signals (timestamp, symbol, signal_type,
               confidence, strategy, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ts, symbol, signal_type, confidence, strategy,
             json.dumps(metadata or {})),
        )
        conn.commit()
        return cursor.lastrowid or 0

    def log_portfolio_snapshot(
        self,
        equity: float,
        cash: float,
        positions: list[dict[str, Any]],
        timestamp: Optional[str] = None,
    ) -> int:
        """Record a portfolio state snapshot."""
        ts = timestamp or datetime.utcnow().isoformat()
        conn = self._get_conn()
        total_exposure = sum(abs(p.get("market_value", 0)) for p in positions)
        cursor = conn.execute(
            """INSERT INTO portfolio_snapshots
               (timestamp, equity, cash, num_positions, positions, total_exposure)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ts, equity, cash, len(positions), json.dumps(positions), total_exposure),
        )
        conn.commit()
        return cursor.lastrowid or 0

    def log_scanner_result(
        self,
        num_scanned: int,
        num_passed: int,
        top_symbols: list[str],
        timestamp: Optional[str] = None,
    ) -> int:
        """Record scanner output."""
        ts = timestamp or datetime.utcnow().isoformat()
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO scanner_results
               (timestamp, num_scanned, num_passed, top_symbols)
               VALUES (?, ?, ?, ?)""",
            (ts, num_scanned, num_passed, json.dumps(top_symbols)),
        )
        conn.commit()
        return cursor.lastrowid or 0

    def log_alert(
        self,
        level: str,
        message: str,
        timestamp: Optional[str] = None,
    ) -> int:
        """Record an alert."""
        ts = timestamp or datetime.utcnow().isoformat()
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO alerts (timestamp, level, message) VALUES (?, ?, ?)",
            (ts, level, message),
        )
        conn.commit()
        return cursor.lastrowid or 0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_trades(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        since: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get recent trades, optionally filtered by symbol."""
        conn = self._get_conn()
        query = "SELECT * FROM trades"
        params: list[Any] = []
        conditions = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_signals(self, limit: int = 100) -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_portfolio_history(self, limit: int = 500) -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_pnl(self, days: int = 30) -> list[dict[str, Any]]:
        """Compute daily P&L from portfolio snapshots."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT DATE(timestamp) as date,
                      MIN(equity) as min_equity,
                      MAX(equity) as max_equity,
                      (SELECT equity FROM portfolio_snapshots p2
                       WHERE DATE(p2.timestamp) = DATE(p.timestamp)
                       ORDER BY p2.timestamp DESC LIMIT 1) as close_equity
               FROM portfolio_snapshots p
               GROUP BY DATE(timestamp)
               ORDER BY date DESC
               LIMIT ?""",
            (days,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_trade_summary(self) -> dict[str, Any]:
        """Get aggregate trade statistics."""
        conn = self._get_conn()
        row = conn.execute(
            """SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN side = 'buy' THEN 1 ELSE 0 END) as buys,
                SUM(CASE WHEN side = 'sell' THEN 1 ELSE 0 END) as sells,
                COUNT(DISTINCT symbol) as unique_symbols,
                AVG(confidence) as avg_confidence,
                MIN(timestamp) as first_trade,
                MAX(timestamp) as last_trade
               FROM trades"""
        ).fetchone()
        return dict(row) if row else {}

    def get_alerts(self, unacknowledged_only: bool = True, limit: int = 50) -> list[dict[str, Any]]:
        conn = self._get_conn()
        query = "SELECT * FROM alerts"
        if unacknowledged_only:
            query += " WHERE acknowledged = 0"
        query += " ORDER BY timestamp DESC LIMIT ?"
        rows = conn.execute(query, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def acknowledge_alert(self, alert_id: int) -> None:
        conn = self._get_conn()
        conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
