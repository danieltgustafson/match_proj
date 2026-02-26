"""Schwab (formerly TD Ameritrade) broker integration stub.

Schwab acquired TD Ameritrade and is transitioning their developer API.
The community-maintained `schwab-py` library provides access, but the
API has been unreliable during the migration.

Status: STUB -- basic structure is in place but not production-ready.
If you want to use Schwab for execution, you'll need to:

1. Apply for API access at https://developer.schwab.com/
2. pip install schwab-py
3. Complete the OAuth flow to get access tokens
4. Fill in the implementation below

For now, Interactive Brokers is the recommended path for your
existing brokerage accounts.

Note: Fidelity does NOT offer a retail trading API, so programmatic
trading through Fidelity is not feasible.
"""

from __future__ import annotations

import logging
from typing import Any

from trading_agent.execution.broker import BaseBroker
from trading_agent.models.signals import Signal

logger = logging.getLogger("trading_agent.schwab")


class SchwabBroker(BaseBroker):
    """Placeholder Schwab broker integration.

    This is a stub -- it logs what it would do but does not submit
    real orders.  See module docstring for how to finish the
    implementation if/when the Schwab API stabilizes.
    """

    def __init__(
        self,
        app_key: str = "",
        app_secret: str = "",
        token_path: str = ".schwab_token.json",
    ) -> None:
        self.app_key = app_key
        self.app_secret = app_secret
        self.token_path = token_path
        logger.warning(
            "SchwabBroker is a stub -- orders will be logged but NOT executed. "
            "Use IBBroker or AlpacaBroker for real execution."
        )

    def submit_order(self, signal: Signal) -> dict[str, Any]:
        logger.info(
            "[STUB] Would submit %s order for %s (confidence=%.2f)",
            signal.signal_type.value,
            signal.symbol,
            signal.confidence,
        )
        return {
            "status": "stub",
            "signal_type": signal.signal_type.value,
            "symbol": signal.symbol,
        }

    def get_positions(self) -> list[dict[str, Any]]:
        logger.info("[STUB] Would fetch Schwab positions")
        return []

    def get_balance(self) -> float:
        logger.info("[STUB] Would fetch Schwab balance")
        return 0.0
