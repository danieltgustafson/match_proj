"""Tests for the Seeking Alpha provider."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from trading_agent.data.providers.seeking_alpha import (
    SAAnalystRating,
    SeekingAlphaError,
    SeekingAlphaProvider,
)


def test_requires_api_key() -> None:
    with pytest.raises(ValueError, match="RapidAPI key"):
        SeekingAlphaProvider(rapidapi_key="")


def test_analyst_rating_compute_sentiment() -> None:
    rating = SAAnalystRating(
        symbol="AAPL",
        sa_authors_rating=4.5,
        wall_street_rating=4.0,
        quant_rating=4.2,
    )
    score = rating.compute_sentiment()
    # Average = 4.23, mapped to (4.23 - 3) / 2 = 0.617
    assert 0.5 < score < 0.7
    assert rating.sentiment_score == score


def test_analyst_rating_neutral() -> None:
    rating = SAAnalystRating(
        sa_authors_rating=3.0,
        wall_street_rating=3.0,
        quant_rating=3.0,
    )
    score = rating.compute_sentiment()
    assert score == 0.0


def test_analyst_rating_bearish() -> None:
    rating = SAAnalystRating(
        sa_authors_rating=1.5,
        wall_street_rating=2.0,
        quant_rating=1.8,
    )
    score = rating.compute_sentiment()
    assert score < -0.3


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


@patch("trading_agent.data.providers.seeking_alpha.httpx.get")
def test_get_ratings(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse({
        "data": {
            "attributes": {
                "authorsRatingPro": 4.2,
                "wallStRating": 3.8,
                "quantRating": 4.5,
            }
        }
    })
    provider = SeekingAlphaProvider(rapidapi_key="test-key")
    rating = provider.get_ratings("AAPL")
    assert rating.sa_authors_rating == 4.2
    assert rating.wall_street_rating == 3.8
    assert rating.quant_rating == 4.5
    assert rating.sentiment_score > 0


@patch("trading_agent.data.providers.seeking_alpha.httpx.get")
def test_get_sentiment_data(mock_get: Any) -> None:
    # First call: ratings, second call: articles
    mock_get.side_effect = [
        FakeResponse({
            "data": {
                "attributes": {
                    "authorsRatingPro": 4.0,
                    "wallStRating": 4.0,
                    "quantRating": 4.0,
                }
            }
        }),
        FakeResponse({
            "data": [
                {"attributes": {"title": "AAPL looks great", "summary": "Buy it"}},
                {"attributes": {"title": "Strong earnings", "teaser": "Beat estimates"}},
            ]
        }),
    ]
    provider = SeekingAlphaProvider(rapidapi_key="test-key")
    sentiment = provider.get_sentiment_data("AAPL")
    assert sentiment.score > 0
    assert sentiment.num_articles == 2
    assert "seeking_alpha" in sentiment.sources


@patch("trading_agent.data.providers.seeking_alpha.httpx.get")
def test_api_error_handled(mock_get: Any) -> None:
    mock_get.return_value = FakeResponse({}, status_code=429)
    provider = SeekingAlphaProvider(rapidapi_key="test-key")
    # Should return default rating, not crash
    rating = provider.get_ratings("AAPL")
    assert rating.sentiment_score == 0.0
