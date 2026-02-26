"""Tests for the news sentiment aggregator."""

from __future__ import annotations

from trading_agent.data.providers.news_sentiment import NewsSentimentAggregator


def test_no_sources_returns_neutral() -> None:
    agg = NewsSentimentAggregator()
    result = agg.get_sentiment("AAPL")
    assert result.score == 0.0
    assert result.num_articles == 0


def test_naive_headline_sentiment_bullish() -> None:
    headlines = [
        "AAPL stock surges on strong earnings beat",
        "Apple rally continues as growth accelerates",
        "Analyst upgrade: Buy Apple stock",
    ]
    score = NewsSentimentAggregator._naive_headline_sentiment(headlines)
    assert score > 0


def test_naive_headline_sentiment_bearish() -> None:
    headlines = [
        "Apple stock plunges on weak guidance",
        "Warning: Risk of further decline in AAPL",
        "Analyst downgrade: Sell Apple stock",
    ]
    score = NewsSentimentAggregator._naive_headline_sentiment(headlines)
    assert score < 0


def test_naive_headline_sentiment_neutral() -> None:
    headlines = [
        "Apple announces new product launch date",
        "Tim Cook speaks at conference",
    ]
    score = NewsSentimentAggregator._naive_headline_sentiment(headlines)
    assert score == 0.0


def test_empty_headlines_neutral() -> None:
    score = NewsSentimentAggregator._naive_headline_sentiment([])
    assert score == 0.0
