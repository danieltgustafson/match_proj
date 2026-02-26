"""Seeking Alpha data provider via RapidAPI.

Seeking Alpha is one of the best sources for:
- Analyst ratings and price targets
- Quant ratings (algorithmic fundamental scores)
- Earnings analysis and estimates
- Community sentiment from articles and comments

SA doesn't offer a direct API, but there are RapidAPI wrappers
that provide structured access to SA data.  The most popular one
is "Seeking Alpha API" on RapidAPI marketplace.

To use this provider:
1. Sign up at https://rapidapi.com/
2. Subscribe to "Seeking Alpha API" (free tier: ~500 req/month)
   URL: https://rapidapi.com/sparsor/api/seeking-alpha-api
3. Set RAPIDAPI_KEY in your .env file

Available data:
- Symbol ratings (SA authors, Wall Street analysts, quant model)
- Recent analysis articles with summaries
- Earnings estimates and surprises
- Dividend information
- Peer comparison metrics

This data feeds into the SentimentStrategy and FactorStrategy
to enhance signal quality with professional analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from trading_agent.strategies.sentiment import SentimentData

logger = logging.getLogger("trading_agent.seeking_alpha")

_RAPIDAPI_HOST = "seeking-alpha-api.p.rapidapi.com"
_BASE_URL = f"https://{_RAPIDAPI_HOST}"


class SeekingAlphaError(Exception):
    """Raised when the Seeking Alpha API returns an error."""


@dataclass
class SAAnalystRating:
    """Structured analyst rating data from Seeking Alpha."""

    symbol: str = ""
    # SA ratings are typically 1-5 (1=strong sell, 5=strong buy)
    sa_authors_rating: float = 3.0      # SA contributor consensus
    wall_street_rating: float = 3.0     # Wall Street analyst consensus
    quant_rating: float = 3.0           # SA quant model score
    # Derived sentiment score in [-1, 1] range
    sentiment_score: float = 0.0

    def compute_sentiment(self) -> float:
        """Convert 1-5 ratings to a [-1, 1] sentiment score.

        Averages all three rating sources and maps:
        1 -> -1.0 (strong sell)
        3 -> 0.0 (hold)
        5 -> +1.0 (strong buy)
        """
        avg = (self.sa_authors_rating + self.wall_street_rating + self.quant_rating) / 3.0
        self.sentiment_score = (avg - 3.0) / 2.0  # maps [1,5] to [-1,1]
        return self.sentiment_score


@dataclass
class SAArticleSummary:
    """Summary of a Seeking Alpha article."""

    title: str = ""
    summary: str = ""
    author: str = ""
    publish_date: str = ""
    sentiment: str = ""  # "bullish", "bearish", "neutral" if available


class SeekingAlphaProvider:
    """Fetch analysis data from Seeking Alpha via RapidAPI.

    Parameters
    ----------
    rapidapi_key:
        Your RapidAPI key.
    """

    def __init__(self, rapidapi_key: str) -> None:
        if not rapidapi_key:
            raise ValueError(
                "RapidAPI key is required for SeekingAlphaProvider. "
                "Sign up at https://rapidapi.com/ and subscribe to "
                "the Seeking Alpha API, then set RAPIDAPI_KEY in .env."
            )
        self.rapidapi_key = rapidapi_key
        self._headers = {
            "x-rapidapi-host": _RAPIDAPI_HOST,
            "x-rapidapi-key": self.rapidapi_key,
        }

    def get_ratings(self, symbol: str) -> SAAnalystRating:
        """Fetch SA author, Wall Street, and quant ratings for a symbol.

        Returns an SAAnalystRating with a computed sentiment_score.
        """
        try:
            data = self._get("/symbols/get-ratings", params={"symbol": symbol})
        except SeekingAlphaError:
            logger.warning("Failed to fetch SA ratings for %s", symbol)
            return SAAnalystRating(symbol=symbol)

        rating = SAAnalystRating(symbol=symbol)

        # Parse the response -- structure varies by RapidAPI wrapper
        if isinstance(data, dict):
            attrs = data.get("data", {}).get("attributes", data)
            rating.sa_authors_rating = float(attrs.get("authorsRatingPro", 3.0) or 3.0)
            rating.wall_street_rating = float(attrs.get("wallStRating", 3.0) or 3.0)
            rating.quant_rating = float(attrs.get("quantRating", 3.0) or 3.0)

        rating.compute_sentiment()
        logger.info(
            "SA ratings for %s: authors=%.1f ws=%.1f quant=%.1f -> sentiment=%.2f",
            symbol,
            rating.sa_authors_rating,
            rating.wall_street_rating,
            rating.quant_rating,
            rating.sentiment_score,
        )
        return rating

    def get_analysis_articles(
        self,
        symbol: str,
        limit: int = 10,
    ) -> list[SAArticleSummary]:
        """Fetch recent analysis articles for a symbol."""
        try:
            data = self._get(
                "/symbols/get-analysis",
                params={"symbol": symbol, "size": str(limit)},
            )
        except SeekingAlphaError:
            logger.warning("Failed to fetch SA articles for %s", symbol)
            return []

        articles = []
        items = data.get("data", []) if isinstance(data, dict) else []
        for item in items[:limit]:
            attrs = item.get("attributes", {})
            articles.append(
                SAArticleSummary(
                    title=attrs.get("title", ""),
                    summary=attrs.get("summary", attrs.get("teaser", "")),
                    author=attrs.get("author", ""),
                    publish_date=attrs.get("publishOn", ""),
                )
            )

        return articles

    def get_sentiment_data(self, symbol: str) -> SentimentData:
        """Get a SentimentData object ready to plug into SentimentStrategy.

        Combines ratings and article data into the standard format.
        """
        rating = self.get_ratings(symbol)
        articles = self.get_analysis_articles(symbol, limit=5)

        return SentimentData(
            score=rating.sentiment_score,
            num_articles=len(articles),
            sources=["seeking_alpha"],
            headlines=[a.title for a in articles if a.title],
        )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict[str, str]] = None) -> Any:
        resp = httpx.get(
            f"{_BASE_URL}{path}",
            headers=self._headers,
            params=params,
            timeout=30.0,
        )
        if resp.status_code >= 400:
            raise SeekingAlphaError(
                f"SA API error ({resp.status_code}): {resp.text[:200]}"
            )
        return resp.json()
