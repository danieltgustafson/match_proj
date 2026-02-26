"""Tests for the SQLite-backed disk cache."""

from __future__ import annotations

import os
import tempfile
import time
from unittest.mock import patch

import pytest

from trading_agent.data.cache import DiskCache


@pytest.fixture()
def cache(tmp_path: object) -> DiskCache:
    """Create a DiskCache in a temporary directory."""
    db_path = os.path.join(str(tmp_path), "test_cache.db")
    return DiskCache(db_path=db_path, ttl=60.0)


def test_set_and_get(cache: DiskCache) -> None:
    cache.set("key1", {"foo": "bar"})
    result = cache.get("key1")
    assert result == {"foo": "bar"}


def test_get_missing_key(cache: DiskCache) -> None:
    assert cache.get("nonexistent") is None


def test_overwrite(cache: DiskCache) -> None:
    cache.set("key1", {"v": 1})
    cache.set("key1", {"v": 2})
    assert cache.get("key1") == {"v": 2}


def test_delete(cache: DiskCache) -> None:
    cache.set("key1", {"v": 1})
    cache.delete("key1")
    assert cache.get("key1") is None


def test_clear(cache: DiskCache) -> None:
    cache.set("a", {"v": 1})
    cache.set("b", {"v": 2})
    cache.clear()
    assert cache.size() == 0


def test_size(cache: DiskCache) -> None:
    assert cache.size() == 0
    cache.set("a", {"v": 1})
    cache.set("b", {"v": 2})
    assert cache.size() == 2


def test_ttl_expiration(tmp_path: object) -> None:
    db_path = os.path.join(str(tmp_path), "ttl_test.db")
    cache = DiskCache(db_path=db_path, ttl=0.5)  # 500ms TTL
    cache.set("key1", {"v": 1})
    assert cache.get("key1") is not None

    time.sleep(0.6)
    assert cache.get("key1") is None  # expired


def test_ttl_zero_keeps_forever(tmp_path: object) -> None:
    db_path = os.path.join(str(tmp_path), "forever.db")
    cache = DiskCache(db_path=db_path, ttl=0)
    cache.set("key1", {"v": 1})
    # With ttl=0, entry should never expire via get()
    assert cache.get("key1") == {"v": 1}


def test_evict_expired(tmp_path: object) -> None:
    db_path = os.path.join(str(tmp_path), "evict.db")
    cache = DiskCache(db_path=db_path, ttl=0.3)
    cache.set("old", {"v": 1})
    time.sleep(0.4)
    cache.set("new", {"v": 2})

    removed = cache.evict_expired()
    assert removed == 1
    assert cache.get("old") is None
    assert cache.get("new") == {"v": 2}


def test_persists_across_instances(tmp_path: object) -> None:
    """Data survives closing and reopening the cache."""
    db_path = os.path.join(str(tmp_path), "persist.db")

    cache1 = DiskCache(db_path=db_path, ttl=3600)
    cache1.set("symbol", {"close": 150.0})
    cache1.close()

    cache2 = DiskCache(db_path=db_path, ttl=3600)
    result = cache2.get("symbol")
    assert result == {"close": 150.0}
    cache2.close()
