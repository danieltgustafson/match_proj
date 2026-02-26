"""Multi-source news sentiment aggregator.

Combines sentiment data from multiple providers into a single score:
- Alpha Vantage NEWS_SENTIMENT endpoint (free, 25 req/day)
- Seeking Alpha ratings and articles (via RapidAPI)
- NewsAPI headlines (optional)

This module acts as the bridge between raw news data and the
SentimentStrategy.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from trading_agent.strategies.sentiment import SentimentData

logger = logging.getLogger("trading_agent.news_sentiment")


class NewsSentimentAggregator:
    """Aggregate sentiment from multiple news/analysis sources.

    Parameters
    ----------
    alpha_vantage_key:
        Alpha Vantage API key (for NEWS_SENTIMENT endpoint).
    rapidapi_key:
        RapidAPI key (for Seeking Alpha data).
    news_api_key:
        NewsAPI.org key (for headline sentiment).
    """

    def __init__(
        self,
        alpha_vantage_key: str = "",
        rapidapi_key: str = "",
        news_api_key: str = "",
    ) -> None:
        self.alpha_vantage_key = alpha_vantage_key
        self.rapidapi_key = rapidapi_key
        self.news_api_key = news_api_key

    def get_sentiment(self, symbol: str) -> SentimentData:
        """Get aggregated sentiment from all configured sources.

        Averages scores across available sources, weighted by
        data quality:
        - Seeking Alpha ratings: weight 0.4 (professional analysis)
        - Alpha Vantage sentiment: weight 0.35 (NLP on news)
        - NewsAPI headlines: weight 0.25 (broad coverage)
        """
        scores: list[tuple[float, float]] = []  # (score, weight)
        all_headlines: list[str] = []
        all_sources: list[str] = []
        total_articles = 0

        # 1. Seeking Alpha (highest quality for fundamentals)
        if self.rapidapi_key:
            try:
                from trading_agent.data.providers.seeking_alpha import (
                    SeekingAlphaProvider,
                )

                sa = SeekingAlphaProvider(self.rapidapi_key)
                sa_data = sa.get_sentiment_data(symbol)
                if sa_data.num_articles > 0 or sa_data.score != 0:
                    scores.append((sa_data.score, 0.4))
                    all_headlines.extend(sa_data.headlines)
                    all_sources.append("seeking_alpha")
                    total_articles += sa_data.num_articles
            except Exception as exc:
                logger.warning("Seeking Alpha fetch failed: %s", exc)

        # 2. Alpha Vantage NEWS_SENTIMENT
        if self.alpha_vantage_key:
            try:
                av_data = self._fetch_alpha_vantage_sentiment(symbol)
                if av_data["score"] is not None:
                    scores.append((av_data["score"], 0.35))
                    all_headlines.extend(av_data["headlines"])
                    all_sources.append("alpha_vantage_news")
                    total_articles += av_data["count"]
            except Exception as exc:
                logger.warning("Alpha Vantage sentiment fetch failed: %s", exc)

        # 3. NewsAPI headlines
        if self.news_api_key:
            try:
                news_data = self._fetch_newsapi_headlines(symbol)
                if news_data["headlines"]:
                    # Simple headline sentiment: count positive/negative keywords
                    score = self._naive_headline_sentiment(news_data["headlines"])
                    scores.append((score, 0.25))
                    all_headlines.extend(news_data["headlines"])
                    all_sources.append("newsapi")
                    total_articles += len(news_data["headlines"])
            except Exception as exc:
                logger.warning("NewsAPI fetch failed: %s", exc)

        # Weighted average
        if not scores:
            return SentimentData(score=0.0, num_articles=0, sources=[], headlines=[])

        total_weight = sum(w for _, w in scores)
        final_score = sum(s * w for s, w in scores) / total_weight

        return SentimentData(
            score=max(-1.0, min(1.0, final_score)),
            num_articles=total_articles,
            sources=all_sources,
            headlines=all_headlines[:20],  # cap at 20
        )

    def _fetch_alpha_vantage_sentiment(self, symbol: str) -> dict[str, Any]:
        """Fetch sentiment from Alpha Vantage NEWS_SENTIMENT endpoint."""
        resp = httpx.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "apikey": self.alpha_vantage_key,
                "limit": "10",
            },
            timeout=30.0,
        )
        data = resp.json()

        if "feed" not in data:
            return {"score": None, "headlines": [], "count": 0}

        articles = data["feed"]
        headlines = [a.get("title", "") for a in articles]

        # Extract ticker-specific sentiment scores
        ticker_scores = []
        for article in articles:
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == symbol.upper():
                    score_str = ts.get("ticker_sentiment_score", "0")
                    try:
                        ticker_scores.append(float(score_str))
                    except ValueError:
                        pass

        avg_score = sum(ticker_scores) / len(ticker_scores) if ticker_scores else 0.0

        return {
            "score": avg_score,
            "headlines": headlines,
            "count": len(articles),
        }

    def _fetch_newsapi_headlines(self, symbol: str) -> dict[str, Any]:
        """Fetch recent headlines from NewsAPI."""
        resp = httpx.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": symbol,
                "sortBy": "publishedAt",
                "pageSize": "10",
                "apiKey": self.news_api_key,
            },
            timeout=30.0,
        )
        data = resp.json()
        articles = data.get("articles", [])
        headlines = [a.get("title", "") for a in articles if a.get("title")]
        return {"headlines": headlines}

    @staticmethod
    def _naive_headline_sentiment(headlines: list[str]) -> float:
        """Very simple keyword-based sentiment scoring.

        This is a placeholder -- for production, use FinBERT or similar.
        Returns a score in [-1, 1].
        """
        positive = {
            "surge", "soar", "jump", "rally", "gain", "beat", "upgrade",
            "bullish", "record", "growth", "profit", "strong", "buy",
            "outperform", "positive", "high", "boom", "breakout",
        }
        negative = {
            "crash", "plunge", "drop", "fall", "miss", "downgrade",
            "bearish", "loss", "weak", "sell", "underperform", "negative",
            "low", "decline", "slump", "cut", "warning", "risk",
        }

        pos_count = 0
        neg_count = 0
        for headline in headlines:
            words = set(headline.lower().split())
            pos_count += len(words & positive)
            neg_count += len(words & negative)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return (pos_count - neg_count) / total
