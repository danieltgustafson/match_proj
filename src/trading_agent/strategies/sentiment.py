"""Sentiment analysis strategy -- trade based on news/social sentiment.

This strategy combines fundamental news sentiment with price action to
generate trading signals.  It's designed to plug into a news/sentiment
data source (e.g. NewsAPI, social media sentiment providers, or an LLM
summarizer).

Research basis:
- Tetlock (2007) "Giving Content to Investor Sentiment" -- media pessimism
  predicts downward pressure on prices.
- Bollen, Mao & Zeng (2011) "Twitter mood predicts the stock market" --
  aggregate social sentiment has predictive power.
- Modern NLP / LLM approaches for earnings call analysis and news
  classification (FinBERT, GPT-based summarizers).

Data sources you will need (pick one or more):
- NewsAPI (https://newsapi.org) -- free tier available
- Alpha Vantage NEWS_SENTIMENT endpoint
- Twitter/X API for social sentiment
- SEC EDGAR for earnings filings (free)

The strategy accepts a pre-computed sentiment score in [-1, 1] range
(negative = bearish, positive = bullish) so the actual NLP pipeline
is decoupled and can use any backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from trading_agent.models.signals import Signal, SignalType
from trading_agent.strategies.base import BaseStrategy


@dataclass
class SentimentData:
    """Pre-computed sentiment payload for a symbol.

    Attributes
    ----------
    score:
        Aggregate sentiment in [-1, 1].  -1 = very bearish, +1 = very bullish.
    num_articles:
        Number of articles / data points used to compute the score.
    sources:
        List of source names (e.g. ["reuters", "bloomberg"]).
    headlines:
        Recent headline strings for context.
    """

    score: float = 0.0
    num_articles: int = 0
    sources: list[str] = field(default_factory=list)
    headlines: list[str] = field(default_factory=list)


class SentimentStrategy(BaseStrategy):
    """Generate signals from news/social sentiment combined with price trend.

    The strategy blends two components:
    1. Raw sentiment score (from NLP pipeline)
    2. Recent price momentum (to avoid fighting strong trends)

    Parameters
    ----------
    bullish_threshold:
        Sentiment score above which we consider a BUY.
    bearish_threshold:
        Sentiment score below which we consider a SELL.
    momentum_window:
        Number of bars for short-term price momentum confirmation.
    require_momentum_confirmation:
        If True, only trade when sentiment and momentum agree.
    """

    def __init__(
        self,
        bullish_threshold: float = 0.3,
        bearish_threshold: float = -0.3,
        momentum_window: int = 5,
        require_momentum_confirmation: bool = True,
    ) -> None:
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.momentum_window = momentum_window
        self.require_momentum_confirmation = require_momentum_confirmation

    def evaluate(self, data: pd.DataFrame) -> Signal:
        """Evaluate sentiment + price data and produce a signal.

        Expects `data` to have a "close" column and optionally a
        "sentiment_score" column.  If sentiment_score is missing,
        pass it via the `sentiment` kwarg in metadata or call
        `evaluate_with_sentiment()` directly.
        """
        sentiment_score = 0.0
        if "sentiment_score" in data.columns and len(data) > 0:
            sentiment_score = float(data["sentiment_score"].iloc[-1])

        return self.evaluate_with_sentiment(data, sentiment_score)

    def evaluate_with_sentiment(
        self,
        data: pd.DataFrame,
        sentiment_score: float,
        sentiment_data: Optional[SentimentData] = None,
    ) -> Signal:
        """Core evaluation with explicit sentiment score.

        Parameters
        ----------
        data:
            OHLCV price data.
        sentiment_score:
            Pre-computed sentiment in [-1, 1].
        sentiment_data:
            Optional detailed sentiment payload for metadata.
        """
        meta: dict[str, Any] = {"sentiment_score": sentiment_score}
        if sentiment_data:
            meta["num_articles"] = sentiment_data.num_articles
            meta["sources"] = sentiment_data.sources

        if data.empty or "close" not in data.columns:
            return Signal(signal_type=SignalType.HOLD, confidence=0.0, metadata=meta)

        # Compute short-term momentum
        momentum = 0.0
        if len(data) >= self.momentum_window + 1:
            recent = data["close"].iloc[-self.momentum_window :]
            momentum = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0]
        meta["price_momentum"] = momentum

        # Bullish case
        if sentiment_score >= self.bullish_threshold:
            if self.require_momentum_confirmation and momentum < 0:
                # Sentiment says buy but price is falling -- stay out
                meta["conflict"] = "sentiment_bullish_momentum_bearish"
                return Signal(
                    signal_type=SignalType.HOLD,
                    confidence=abs(sentiment_score) * 0.3,
                    strategy_name="sentiment",
                    metadata=meta,
                )
            confidence = min(abs(sentiment_score), 1.0)
            return Signal(
                signal_type=SignalType.BUY,
                confidence=confidence,
                strategy_name="sentiment",
                metadata=meta,
            )

        # Bearish case
        if sentiment_score <= self.bearish_threshold:
            if self.require_momentum_confirmation and momentum > 0:
                meta["conflict"] = "sentiment_bearish_momentum_bullish"
                return Signal(
                    signal_type=SignalType.HOLD,
                    confidence=abs(sentiment_score) * 0.3,
                    strategy_name="sentiment",
                    metadata=meta,
                )
            confidence = min(abs(sentiment_score), 1.0)
            return Signal(
                signal_type=SignalType.SELL,
                confidence=confidence,
                strategy_name="sentiment",
                metadata=meta,
            )

        return Signal(
            signal_type=SignalType.HOLD,
            confidence=0.0,
            strategy_name="sentiment",
            metadata=meta,
        )
