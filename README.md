# Trading Agent

A Python-centric framework for building automated stock and cryptocurrency trading agents.

## Overview

This project provides a modular, extensible architecture for:

- **Market data ingestion** from multiple sources (Yahoo Finance, crypto exchanges)
- **Technical indicator computation** (RSI, MACD, Bollinger Bands, etc.)
- **Strategy development** with a clean base class pattern
- **Agent orchestration** that ties data, signals, and execution together
- **Paper trading** for safe backtesting before going live
- **Broker integration** for live order execution

## Project Structure

```
src/trading_agent/
    agents/          # Agent orchestrators (stock, crypto)
    strategies/      # Trading strategies (momentum, mean reversion, etc.)
    data/
        providers/   # Market data sources
        processors.py
    models/          # Signal models and ML integration points
    execution/       # Broker and paper trading execution
    utils/           # Indicators, logging, helpers
tests/               # Test suite
```

## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/danieltgustafson/match_proj.git
cd match_proj

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Configuration

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

### Running Tests

```bash
pytest
```

### Usage

```python
from trading_agent.config import Settings
from trading_agent.agents.stock_agent import StockAgent
from trading_agent.strategies.momentum import MomentumStrategy

settings = Settings()
strategy = MomentumStrategy(rsi_period=14, rsi_overbought=70, rsi_oversold=30)
agent = StockAgent(settings=settings, strategy=strategy)

# Run a backtest
results = agent.backtest(symbol="AAPL", start="2024-01-01", end="2024-12-31")
print(results.summary())
```

## Development

```bash
# Run linting
ruff check src/ tests/

# Run type checking
mypy src/

# Run tests with coverage
pytest --cov=trading_agent
```

## License

MIT
