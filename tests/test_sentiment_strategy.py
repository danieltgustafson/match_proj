"""Tests for the sentiment analysis strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_agent.models.signals import SignalType
from trading_agent.strategies.sentiment import SentimentData, SentimentStrategy


def _make_price_data(prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": prices})


def test_bullish_sentiment_buy() -> None:
    strategy = SentimentStrategy(require_momentum_confirmation=False)
    data = _make_price_data([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    signal = strategy.evaluate_with_sentiment(data, sentiment_score=0.6)
    assert signal.signal_type is SignalType.BUY
    assert signal.confidence > 0


def test_bearish_sentiment_sell() -> None:
    strategy = SentimentStrategy(require_momentum_confirmation=False)
    data = _make_price_data([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])
    signal = strategy.evaluate_with_sentiment(data, sentiment_score=-0.5)
    assert signal.signal_type is SignalType.SELL


def test_neutral_sentiment_hold() -> None:
    strategy = SentimentStrategy()
    data = _make_price_data([100.0, 100.0, 100.0])
    signal = strategy.evaluate_with_sentiment(data, sentiment_score=0.1)
    assert signal.signal_type is SignalType.HOLD


def test_conflicting_sentiment_and_momentum_holds() -> None:
    strategy = SentimentStrategy(require_momentum_confirmation=True)
    # Bullish sentiment but falling prices
    data = _make_price_data([105.0, 104.0, 103.0, 102.0, 101.0, 100.0])
    signal = strategy.evaluate_with_sentiment(data, sentiment_score=0.5)
    assert signal.signal_type is SignalType.HOLD
    assert "conflict" in signal.metadata


def test_empty_data_hold() -> None:
    strategy = SentimentStrategy()
    signal = strategy.evaluate_with_sentiment(pd.DataFrame(), sentiment_score=0.8)
    assert signal.signal_type is SignalType.HOLD


def test_sentiment_data_in_metadata() -> None:
    strategy = SentimentStrategy(require_momentum_confirmation=False)
    data = _make_price_data([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    sd = SentimentData(
        score=0.7,
        num_articles=15,
        sources=["reuters", "bloomberg"],
    )
    signal = strategy.evaluate_with_sentiment(data, sentiment_score=0.7, sentiment_data=sd)
    assert signal.metadata["num_articles"] == 15
