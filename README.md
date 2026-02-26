# Trading Agent

A Python framework for building automated stock and cryptocurrency trading agents. Combines algorithmic technical analysis with fundamental/factor-based strategies, all orchestrated through a risk-managed execution pipeline that actually submits orders.

## Architecture

```
Signal Generation          Position Sizing          Risk Checks          Execution
  (Strategies)           (Portfolio Manager)       (Risk Manager)        (Broker)
                                                                    
  Momentum (RSI)  ─┐                                                  ┌─ Alpaca (live)
  Mean Reversion  ─┤                                                  │
  MACD Crossover  ─┼─> Ensemble ──> Kelly / Risk ──> Drawdown  ──────>├─ Paper (sim)
  Sentiment/News  ─┤     Merge      Parity Sizing    Daily Loss       │
  Factor/Fundmtl  ─┘                                 Concentration    └─ (your broker)
```

### Key Design Decisions

- **Multi-strategy ensemble**: Multiple independent alpha sources are combined via weighted voting. This is a well-established approach for improving Sharpe ratio (reduces variance while preserving signal).
- **Kelly Criterion sizing**: Position sizes are computed using the Kelly formula (with a configurable fractional Kelly, default half-Kelly, to reduce variance). References: Kelly (1956), Thorp (2006).
- **Risk parity alternative**: Optionally size positions so each contributes equal volatility risk, based on Bridgewater's All Weather approach.
- **Circuit breakers**: The risk manager halts trading if drawdown exceeds a threshold (default 15%) or daily losses exceed a cap (default 3%). Sell orders are always allowed through.
- **Decoupled sentiment pipeline**: The sentiment strategy accepts a pre-computed score in [-1, 1], so you can plug in any NLP backend (FinBERT, GPT-based, NewsAPI, etc.) without touching the trading logic.
- **Factor model**: Multi-factor scoring (value, quality, momentum, low-vol) based on Fama-French and related academic research.

## Project Structure

```
src/trading_agent/
    runner.py               # Main orchestrator -- the trading loop
    config.py               # Settings loaded from .env
    __main__.py             # CLI entry point
    agents/                 # Agent wrappers (stock, crypto)
    strategies/
        momentum.py         # RSI-based momentum
        mean_reversion.py   # Bollinger Band mean reversion
        macd_crossover.py   # MACD line crossover
        sentiment.py        # News/social sentiment signals
        factor.py           # Fundamental factor model
        ensemble.py         # Multi-strategy combiner
    data/
        providers/
            alpha_vantage.py  # Stock + crypto OHLCV data
            yahoo_finance.py  # Yahoo Finance wrapper
        processors.py         # Data cleaning, returns
    execution/
        alpaca_broker.py      # Live order execution via Alpaca
        paper_trading.py      # Simulated broker for testing
        portfolio_manager.py  # Position sizing (Kelly, risk parity)
        risk_manager.py       # Drawdown limits, exposure checks
    models/
        signals.py            # Signal, Order, Position models
    utils/
        indicators.py         # RSI, SMA, EMA, Bollinger, MACD
        logging.py
tests/                        # Full test suite
```

## Quick Start

### 1. Install

```bash
git clone https://github.com/danieltgustafson/match_proj.git
cd match_proj
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys (see "Data Sources" below)
```

### 3. Run (paper trading)

```bash
# Single pass -- evaluate watchlist and print signals
python -m trading_agent

# With custom symbols
python -m trading_agent --symbols AAPL,MSFT,NVDA,BTC

# Scheduled execution (every 15 min)
python -m trading_agent --schedule

# Custom interval (every 5 min, max 10 runs)
python -m trading_agent --schedule --interval 300 --max-runs 10
```

### 4. Run (live trading)

```bash
# CAUTION: This submits real orders to Alpaca
python -m trading_agent --live
```

### Programmatic Usage

```python
from trading_agent.config import Settings
from trading_agent.runner import TradingRunner
from trading_agent.strategies.ensemble import EnsembleStrategy
from trading_agent.strategies.momentum import MomentumStrategy
from trading_agent.strategies.sentiment import SentimentStrategy
from trading_agent.strategies.factor import FactorStrategy

# Build a custom ensemble
strategy = EnsembleStrategy(
    strategies={
        "momentum": MomentumStrategy(rsi_period=14),
        "sentiment": SentimentStrategy(bullish_threshold=0.3),
        "factor": FactorStrategy(),
    },
    weights={"momentum": 0.4, "sentiment": 0.3, "factor": 0.3},
    min_agreement=0.5,
)

runner = TradingRunner.from_settings(
    watchlist=["AAPL", "MSFT", "GOOGL"],
    strategy=strategy,
)
result = runner.run_once()
print(result.summary())
```

## Data Sources You Will Need

| Source | What It Provides | Cost | Required? |
|--------|-----------------|------|-----------|
| **Alpha Vantage** | Stock/crypto OHLCV, fundamentals | Free (25 req/day) or $50/mo premium | Yes (already configured) |
| **Alpaca** | Order execution, real-time data | Free (paper), free (live with account) | Yes for live trading |
| **NewsAPI** | News headlines for sentiment | Free (100 req/day) | Optional (for sentiment strategy) |
| **Yahoo Finance** | OHLCV backup data source | Free | Optional (fallback) |

### Getting API Keys

1. **Alpha Vantage**: https://www.alphavantage.co/support/#api-key (already set up)
2. **Alpaca**: https://app.alpaca.markets/signup -- sign up for a free paper trading account
3. **NewsAPI**: https://newsapi.org/register -- free developer tier

## Position Sizing Methods

| Method | Description | When to Use |
|--------|-------------|-------------|
| `kelly` | Kelly Criterion (default half-Kelly) | Best risk-adjusted sizing; use when you have edge estimates |
| `risk_parity` | Size inversely proportional to volatility | When you want equal risk contribution per position |
| `fixed_fractional` | Risk a flat % of equity per trade | Simplest; good starting point |

Configure in `.env`:
```
SIZING_METHOD=kelly
KELLY_FRACTION=0.5
```

## Risk Management

The risk manager enforces hard limits before any order reaches the broker:

- **Max drawdown**: Halts all trading if portfolio drops >15% from peak
- **Daily loss limit**: Stops trading for the day if losses exceed 3%
- **Position concentration**: No single holding >10% of equity
- **Total exposure**: Caps total market exposure at 100% of equity
- **Max positions**: Enforces diversification (default: 20 positions)
- **Restricted symbols**: Blacklist specific tickers

All thresholds are configurable in `.env`.

## Development

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=trading_agent --cov-report=term-missing

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Research References

- Kelly, J.L. (1956) "A New Interpretation of Information Rate" -- optimal bet sizing
- Fama & French (1992) "The Cross-Section of Expected Stock Returns" -- value factor
- Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers" -- momentum
- Novy-Marx (2013) "The Other Side of Value" -- quality/profitability factor
- Thorp, E.O. (2006) "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
- Tetlock (2007) "Giving Content to Investor Sentiment" -- news sentiment
- Bollen, Mao & Zeng (2011) "Twitter mood predicts the stock market"
- Maillard, Roncalli & Teiletche (2010) "The Properties of Equally Weighted Risk Contribution Portfolios" -- risk parity

## License

MIT
