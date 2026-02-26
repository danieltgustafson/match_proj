"""Disk-backed response cache using SQLite.

Persists API responses across process restarts so you don't waste
your free-tier Alpha Vantage quota re-fetching the same data.

Usage::

    cache = DiskCache(db_path="~/.trading_agent/cache.db", ttl=3600)
    cache.set("AAPL|daily|compact", {"Time Series (Daily)": {...}})
    hit = cache.get("AAPL|daily|compact")  # returns dict or None
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("trading_agent.cache")

_DEFAULT_DB_DIR = os.path.join(Path.home(), ".trading_agent")
_DEFAULT_DB_PATH = os.path.join(_DEFAULT_DB_DIR, "cache.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS response_cache (
    cache_key   TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,
    cached_at   REAL NOT NULL
)
"""

_UPSERT = """
INSERT INTO response_cache (cache_key, payload, cached_at)
VALUES (?, ?, ?)
ON CONFLICT(cache_key)
DO UPDATE SET payload = excluded.payload, cached_at = excluded.cached_at
"""

_SELECT = """
SELECT payload, cached_at FROM response_cache WHERE cache_key = ?
"""

_DELETE_EXPIRED = """
DELETE FROM response_cache WHERE cached_at < ?
"""

_DELETE_KEY = """
DELETE FROM response_cache WHERE cache_key = ?
"""

_DELETE_ALL = """
DELETE FROM response_cache
"""

_COUNT = """
SELECT COUNT(*) FROM response_cache
"""


class DiskCache:
    """SQLite-backed key/value cache with TTL expiration.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Parent directories are
        created automatically.  Defaults to ``~/.trading_agent/cache.db``.
    ttl:
        Time-to-live in seconds for cached entries.  Default 3600 (1 hour).
        Set to 0 to keep entries forever (manual eviction only).
    """

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        ttl: float = 3600.0,
    ) -> None:
        self.db_path = db_path
        self.ttl = ttl

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Return cached payload for *key*, or None if missing / expired."""
        row = self._conn.execute(_SELECT, (key,)).fetchone()
        if row is None:
            return None

        payload_json, cached_at = row

        if self.ttl > 0 and (time.time() - cached_at) > self.ttl:
            self._conn.execute(_DELETE_KEY, (key,))
            self._conn.commit()
            logger.debug("Cache expired for key=%s", key)
            return None

        logger.debug("Disk cache hit for key=%s", key)
        result: dict[str, Any] = json.loads(payload_json)
        return result

    def set(self, key: str, payload: dict[str, Any]) -> None:
        """Store *payload* under *key* with the current timestamp."""
        payload_json = json.dumps(payload)
        self._conn.execute(_UPSERT, (key, payload_json, time.time()))
        self._conn.commit()
        logger.debug("Cached key=%s (%d bytes)", key, len(payload_json))

    def delete(self, key: str) -> None:
        """Remove a single entry."""
        self._conn.execute(_DELETE_KEY, (key,))
        self._conn.commit()

    def clear(self) -> None:
        """Remove all cached entries."""
        self._conn.execute(_DELETE_ALL)
        self._conn.commit()
        logger.info("Disk cache cleared")

    def evict_expired(self) -> int:
        """Delete all entries older than TTL.  Returns count of removed rows."""
        if self.ttl <= 0:
            return 0
        cutoff = time.time() - self.ttl
        cursor = self._conn.execute(_DELETE_EXPIRED, (cutoff,))
        self._conn.commit()
        removed = cursor.rowcount
        if removed:
            logger.info("Evicted %d expired cache entries", removed)
        return removed

    def size(self) -> int:
        """Return number of entries in cache (including expired)."""
        row = self._conn.execute(_COUNT).fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
