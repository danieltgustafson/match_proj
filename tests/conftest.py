"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """Return a small synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.5,
            "high": close + abs(np.random.randn(n)),
            "low": close - abs(np.random.randn(n)),
            "close": close,
            "volume": np.random.randint(1_000, 100_000, size=n),
        },
        index=dates,
    )
