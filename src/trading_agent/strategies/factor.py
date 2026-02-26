"""Factor-based (fundamental) strategy.

Implements a multi-factor scoring model inspired by academic factor
investing research:

- **Value** (Fama-French, 1992): Low P/E, low P/B stocks outperform.
- **Quality** (Novy-Marx, 2013): High gross profitability predicts returns.
- **Momentum** (Jegadeesh & Titman, 1993): Past winners continue winning.
- **Low Volatility** (Baker, Bradley & Wurgler, 2011): Less volatile stocks
  deliver higher risk-adjusted returns (the "low vol anomaly").

Each factor produces a z-score, and the composite signal is a weighted
average.  Symbols scoring above/below configurable thresholds trigger
BUY/SELL signals.

Data requirements:
    The strategy needs fundamental + price data per symbol.  You can
    source this from:
    - Alpha Vantage OVERVIEW endpoint (P/E, P/B, EPS, profit margins)
    - Yahoo Finance (via yfinance.Ticker.info)
    - SEC EDGAR XBRL filings (free, comprehensive)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from trading_agent.models.signals import Signal, SignalType
from trading_agent.strategies.base import BaseStrategy


@dataclass
class FundamentalData:
    """Fundamental metrics for a single symbol.

    All optional -- the strategy scores based on whatever is available.
    """

    symbol: str = ""
    pe_ratio: Optional[float] = None        # Price / Earnings
    pb_ratio: Optional[float] = None        # Price / Book
    gross_margin: Optional[float] = None    # Gross profit / Revenue
    roe: Optional[float] = None             # Return on Equity
    debt_to_equity: Optional[float] = None
    revenue_growth: Optional[float] = None  # YoY revenue growth
    earnings_growth: Optional[float] = None # YoY earnings growth
    dividend_yield: Optional[float] = None

    # Price-based (can compute from OHLCV)
    momentum_12m: Optional[float] = None    # 12-month return
    volatility_daily: Optional[float] = None


@dataclass
class FactorWeights:
    """Weights for each factor in the composite score."""

    value: float = 0.25
    quality: float = 0.25
    momentum: float = 0.30
    low_volatility: float = 0.20


class FactorStrategy(BaseStrategy):
    """Multi-factor fundamental strategy.

    Scores each symbol on value, quality, momentum, and volatility
    factors, then generates BUY/SELL/HOLD signals based on composite
    z-scores.

    For single-symbol evaluation (via the standard `evaluate` interface),
    pass fundamental data as columns in the DataFrame or use
    `evaluate_fundamentals()` directly.

    For cross-sectional ranking (comparing multiple stocks), use
    `rank_universe()` which produces signals for a whole watchlist.

    Parameters
    ----------
    weights:
        Relative importance of each factor.
    buy_threshold:
        Composite z-score above which we BUY.
    sell_threshold:
        Composite z-score below which we SELL.
    """

    def __init__(
        self,
        weights: Optional[FactorWeights] = None,
        buy_threshold: float = 0.5,
        sell_threshold: float = -0.5,
    ) -> None:
        self.weights = weights or FactorWeights()
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def evaluate(self, data: pd.DataFrame) -> Signal:
        """Standard interface -- extract fundamentals from DataFrame columns."""
        if data.empty:
            return Signal(signal_type=SignalType.HOLD, confidence=0.0)

        fd = FundamentalData()
        last = data.iloc[-1]
        for attr in ["pe_ratio", "pb_ratio", "gross_margin", "roe",
                      "debt_to_equity", "revenue_growth", "earnings_growth"]:
            if attr in data.columns:
                val = last[attr]
                if pd.notna(val):
                    setattr(fd, attr, float(val))

        # Compute momentum from price data
        if "close" in data.columns and len(data) > 1:
            fd.momentum_12m = (
                (data["close"].iloc[-1] - data["close"].iloc[0])
                / data["close"].iloc[0]
            )
            if len(data) >= 20:
                fd.volatility_daily = float(data["close"].pct_change().std())

        return self.evaluate_fundamentals(fd)

    def evaluate_fundamentals(self, fd: FundamentalData) -> Signal:
        """Score a single symbol using its fundamental data."""
        scores: dict[str, float] = {}
        weights_used: dict[str, float] = {}

        # Value factor: lower P/E and P/B is better
        value_scores = []
        if fd.pe_ratio is not None and fd.pe_ratio > 0:
            # Invert: lower P/E -> higher score
            value_scores.append(-fd.pe_ratio / 30.0)  # normalize around 30
        if fd.pb_ratio is not None and fd.pb_ratio > 0:
            value_scores.append(-fd.pb_ratio / 5.0)
        if value_scores:
            scores["value"] = float(np.mean(value_scores))
            weights_used["value"] = self.weights.value

        # Quality factor: higher margins and ROE is better
        quality_scores = []
        if fd.gross_margin is not None:
            quality_scores.append(fd.gross_margin)
        if fd.roe is not None:
            quality_scores.append(fd.roe)
        if fd.earnings_growth is not None:
            quality_scores.append(fd.earnings_growth)
        if quality_scores:
            scores["quality"] = float(np.mean(quality_scores))
            weights_used["quality"] = self.weights.quality

        # Momentum factor
        if fd.momentum_12m is not None:
            scores["momentum"] = fd.momentum_12m
            weights_used["momentum"] = self.weights.momentum

        # Low volatility factor (lower vol -> higher score)
        if fd.volatility_daily is not None and fd.volatility_daily > 0:
            scores["low_vol"] = -fd.volatility_daily * 10  # scale up
            weights_used["low_volatility"] = self.weights.low_volatility

        if not scores:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                strategy_name="factor",
                metadata={"reason": "insufficient data"},
            )

        # Weighted composite
        total_weight = sum(weights_used.values())
        if total_weight == 0:
            return Signal(signal_type=SignalType.HOLD, confidence=0.0, strategy_name="factor")

        composite = 0.0
        for factor_name, score in scores.items():
            w_key = "low_volatility" if factor_name == "low_vol" else factor_name
            w = weights_used.get(w_key, 0.0)
            composite += score * (w / total_weight)

        meta: dict[str, Any] = {
            "composite_score": composite,
            "factor_scores": scores,
            "symbol": fd.symbol,
        }

        if composite >= self.buy_threshold:
            confidence = min(abs(composite), 1.0)
            return Signal(
                signal_type=SignalType.BUY,
                symbol=fd.symbol,
                confidence=confidence,
                strategy_name="factor",
                metadata=meta,
            )

        if composite <= self.sell_threshold:
            confidence = min(abs(composite), 1.0)
            return Signal(
                signal_type=SignalType.SELL,
                symbol=fd.symbol,
                confidence=confidence,
                strategy_name="factor",
                metadata=meta,
            )

        return Signal(
            signal_type=SignalType.HOLD,
            symbol=fd.symbol,
            confidence=0.0,
            strategy_name="factor",
            metadata=meta,
        )

    def rank_universe(
        self,
        fundamentals: list[FundamentalData],
    ) -> list[Signal]:
        """Score and rank an entire universe of symbols.

        Returns a list of Signals sorted by composite score (best first).
        Useful for portfolio construction -- buy the top N, sell the bottom N.
        """
        signals = [self.evaluate_fundamentals(fd) for fd in fundamentals]
        # Sort by composite score descending
        signals.sort(
            key=lambda s: s.metadata.get("composite_score", 0),
            reverse=True,
        )
        return signals
