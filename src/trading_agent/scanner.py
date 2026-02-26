"""Universe scanner -- autonomously discover tradeable symbols.

Instead of manually maintaining a watchlist, the scanner:

1. Starts from a broad universe (all US-listed stocks, S&P 500,
   or a configurable index)
2. Applies filters to narrow down to actionable candidates:
   - Minimum volume (liquidity)
   - Price range
   - Momentum / trend strength
   - Volatility (not too low, not too high)
   - Sector diversification
3. Returns a ranked list of symbols for the runner to evaluate

Data sources:
- Alpha Vantage LISTING_STATUS (free, gives all US tickers)
- Yahoo Finance for screening metrics (no rate limit)

The scanner runs before each trading cycle and produces a fresh
watchlist, so the agent is always looking at the most relevant
opportunities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_agent.scanner")


# Built-in universe seeds -- common index constituents
# These are starting points; the scanner filters down from here

SP500_SAMPLE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
    "UNH", "JNJ", "V", "XOM", "JPM", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
    "ACN", "ABT", "TMO", "DHR", "CRM", "NKE", "TXN", "NEE", "PM",
    "UNP", "RTX", "HON", "INTC", "QCOM", "LOW", "AMGN", "IBM", "CAT",
    "BA", "GE", "AMAT", "SBUX", "DE", "ADP", "MDLZ", "GILD", "ADI",
    "MMC", "ISRG", "PLD", "BKNG", "VRTX", "REGN", "LRCX", "SYK",
]

GROWTH_TECH = [
    "PLTR", "SNOW", "DDOG", "NET", "CRWD", "ZS", "MDB", "PANW",
    "COIN", "SQ", "SOFI", "HOOD", "AFRM", "RBLX", "U", "TTD",
    "BILL", "HUBS", "VEEV", "DKNG", "ABNB", "DASH", "PINS", "SNAP",
]

CRYPTO_RELATED = [
    "COIN", "MARA", "RIOT", "CLSK", "HUT", "BITF", "MSTR", "GLXY",
]

SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU",
    "XLRE", "XLC", "SPY", "QQQ", "IWM", "DIA",
]

ALL_SEEDS = list(set(SP500_SAMPLE + GROWTH_TECH + CRYPTO_RELATED + SECTOR_ETFS))


@dataclass
class ScannerConfig:
    """Screening criteria for the universe scanner."""

    # Starting universe: "sp500", "growth", "crypto", "all", or a custom list
    universe: str = "all"

    # Minimum average daily volume (shares)
    min_avg_volume: int = 500_000

    # Price range
    min_price: float = 5.0
    max_price: float = 10_000.0

    # Minimum 20-day average dollar volume (liquidity filter)
    min_dollar_volume: float = 10_000_000.0

    # Momentum filters
    min_momentum_20d: float = -0.50   # reject if dropped more than 50% in 20d
    max_momentum_20d: float = 2.0     # reject if up more than 200% (likely squeeze)

    # Volatility filters (daily return std dev)
    min_volatility: float = 0.005     # reject if too boring (no opportunity)
    max_volatility: float = 0.10      # reject if too wild (too much risk)

    # How many symbols to return (top N after ranking)
    max_symbols: int = 30

    # Ranking criteria: "momentum", "volume", "volatility", "composite"
    rank_by: str = "composite"

    # Custom seed list (overrides universe setting if provided)
    custom_symbols: list[str] = field(default_factory=list)


class UniverseScanner:
    """Scan the market and produce a dynamic watchlist.

    Usage::

        scanner = UniverseScanner()
        watchlist = scanner.scan()  # returns list of symbol strings
        runner = TradingRunner.from_settings(watchlist=watchlist)
    """

    def __init__(self, config: Optional[ScannerConfig] = None) -> None:
        self.config = config or ScannerConfig()

    def scan(self) -> list[str]:
        """Run the full scan pipeline and return ranked symbols.

        Steps:
        1. Get the seed universe
        2. Fetch screening data for each symbol
        3. Apply filters
        4. Rank and return top N
        """
        seeds = self._get_seed_universe()
        logger.info("Scanning %d seed symbols...", len(seeds))

        # Fetch data in bulk via yfinance (much faster than per-symbol)
        screen_data = self._fetch_screening_data(seeds)
        if screen_data.empty:
            logger.warning("No screening data returned, falling back to seeds")
            return seeds[: self.config.max_symbols]

        # Apply filters
        filtered = self._apply_filters(screen_data)
        logger.info(
            "Filtered %d -> %d symbols", len(screen_data), len(filtered)
        )

        if filtered.empty:
            logger.warning("All symbols filtered out, returning top seeds")
            return seeds[: self.config.max_symbols]

        # Rank
        ranked = self._rank(filtered)
        result = ranked.index.tolist()[: self.config.max_symbols]
        logger.info("Scanner result: %s", result)
        return result

    def scan_with_details(self) -> pd.DataFrame:
        """Like scan() but returns the full DataFrame with all metrics.

        Useful for debugging or building a dashboard.
        """
        seeds = self._get_seed_universe()
        screen_data = self._fetch_screening_data(seeds)
        if screen_data.empty:
            return screen_data
        filtered = self._apply_filters(screen_data)
        return self._rank(filtered)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _get_seed_universe(self) -> list[str]:
        """Determine the starting set of symbols to screen."""
        if self.config.custom_symbols:
            return self.config.custom_symbols

        universe = self.config.universe.lower()
        if universe == "sp500":
            return SP500_SAMPLE
        elif universe == "growth":
            return GROWTH_TECH
        elif universe == "crypto":
            return CRYPTO_RELATED
        elif universe == "etfs":
            return SECTOR_ETFS
        else:  # "all"
            return ALL_SEEDS

    def _fetch_screening_data(self, symbols: list[str]) -> pd.DataFrame:
        """Fetch price/volume data for screening.

        Uses yfinance bulk download for speed.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance required for scanner. pip install yfinance")
            return pd.DataFrame()

        try:
            # Download 30 days of data for all symbols at once
            data = yf.download(
                symbols,
                period="1mo",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            logger.error("yfinance download failed: %s", exc)
            return pd.DataFrame()

        if data.empty:
            return pd.DataFrame()

        results = []
        for sym in symbols:
            try:
                if len(symbols) == 1:
                    sym_data = data
                else:
                    sym_data = data[sym] if sym in data.columns.get_level_values(0) else None

                if sym_data is None or sym_data.empty:
                    continue

                sym_data = sym_data.dropna(subset=["Close"])
                if len(sym_data) < 5:
                    continue

                close = sym_data["Close"]
                volume = sym_data["Volume"]

                last_price = float(close.iloc[-1])
                avg_volume = float(volume.mean())
                avg_dollar_volume = last_price * avg_volume

                # 20-day momentum (or whatever data we have)
                momentum = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]

                # Daily volatility
                returns = close.pct_change().dropna()
                volatility = float(returns.std()) if len(returns) > 1 else 0.0

                # Trend strength: abs momentum / volatility
                trend_strength = abs(momentum) / (volatility * np.sqrt(len(close)) + 1e-9)

                results.append({
                    "symbol": sym,
                    "last_price": last_price,
                    "avg_volume": avg_volume,
                    "avg_dollar_volume": avg_dollar_volume,
                    "momentum_20d": float(momentum),
                    "volatility": volatility,
                    "trend_strength": trend_strength,
                })

            except Exception as exc:
                logger.debug("Skipping %s: %s", sym, exc)
                continue

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results).set_index("symbol")
        return df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply screening filters to narrow the universe."""
        c = self.config
        mask = (
            (df["avg_volume"] >= c.min_avg_volume)
            & (df["last_price"] >= c.min_price)
            & (df["last_price"] <= c.max_price)
            & (df["avg_dollar_volume"] >= c.min_dollar_volume)
            & (df["momentum_20d"] >= c.min_momentum_20d)
            & (df["momentum_20d"] <= c.max_momentum_20d)
            & (df["volatility"] >= c.min_volatility)
            & (df["volatility"] <= c.max_volatility)
        )
        return df[mask]

    def _rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank symbols by the configured criteria."""
        if df.empty:
            return df

        rank_by = self.config.rank_by.lower()

        if rank_by == "momentum":
            df = df.sort_values("momentum_20d", ascending=False)
        elif rank_by == "volume":
            df = df.sort_values("avg_dollar_volume", ascending=False)
        elif rank_by == "volatility":
            df = df.sort_values("volatility", ascending=False)
        else:
            # Composite: normalize and combine momentum, volume, trend strength
            for col in ["momentum_20d", "avg_dollar_volume", "trend_strength"]:
                col_min = df[col].min()
                col_range = df[col].max() - col_min
                if col_range > 0:
                    df[f"{col}_norm"] = (df[col] - col_min) / col_range
                else:
                    df[f"{col}_norm"] = 0.0

            df["composite_score"] = (
                0.35 * df.get("momentum_20d_norm", 0)
                + 0.35 * df.get("trend_strength_norm", 0)
                + 0.30 * df.get("avg_dollar_volume_norm", 0)
            )
            df = df.sort_values("composite_score", ascending=False)

        return df
