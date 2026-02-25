"""Stock-market trading agent."""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_agent.agents.base import BaseAgent
from trading_agent.config import Settings
from trading_agent.data.providers.yahoo_finance import YahooFinanceProvider
from trading_agent.execution.paper_trading import PaperBroker
from trading_agent.models.signals import Signal
from trading_agent.strategies.base import BaseStrategy


class StockAgent(BaseAgent):
    """Agent specialised for equity / stock trading."""

    def __init__(self, settings: Settings, strategy: BaseStrategy) -> None:
        super().__init__(settings, strategy)
        self.data_provider = YahooFinanceProvider()
        self.broker = PaperBroker(initial_cash=100_000.0)

    def fetch_data(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        start = kwargs.get("start")
        end = kwargs.get("end")
        return self.data_provider.get_historical(symbol, start=start, end=end)

    def execute_signal(self, signal: Signal) -> dict[str, Any]:
        return self.broker.submit_order(signal)
