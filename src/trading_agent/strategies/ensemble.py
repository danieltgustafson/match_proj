"""Multi-strategy ensemble -- combine signals from multiple strategies.

Research basis:
- Combining independent alpha sources reduces variance and improves
  Sharpe ratio (fundamental result from portfolio theory).
- "Wisdom of crowds" -- aggregating diverse models typically outperforms
  any single model (Surowiecki, 2004; also well-established in ML
  ensemble literature: bagging, boosting, stacking).

The ensemble evaluates each sub-strategy independently, then merges
their signals using configurable weighting schemes:

- **Equal weight**: Each strategy gets the same vote.
- **Confidence-weighted**: Weight by each signal's confidence score.
- **Custom weights**: Manually assign weights per strategy.

The final signal direction is determined by weighted vote, and the
final confidence is the weighted average of individual confidences.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from trading_agent.models.signals import Signal, SignalType
from trading_agent.strategies.base import BaseStrategy

logger = logging.getLogger("trading_agent.ensemble")


class EnsembleStrategy(BaseStrategy):
    """Combine multiple strategies into a single signal.

    Parameters
    ----------
    strategies:
        Dict mapping strategy name -> strategy instance.
    weights:
        Optional dict mapping strategy name -> weight.
        If None, all strategies are weighted equally.
    min_agreement:
        Minimum fraction of strategies that must agree on direction
        for the ensemble to produce an actionable signal (0.0 to 1.0).
        Default 0.5 = simple majority.
    weighting_mode:
        "equal", "confidence", or "custom".
    """

    def __init__(
        self,
        strategies: dict[str, BaseStrategy],
        weights: Optional[dict[str, float]] = None,
        min_agreement: float = 0.5,
        weighting_mode: str = "confidence",
    ) -> None:
        if not strategies:
            raise ValueError("Ensemble requires at least one strategy")
        self.strategies = strategies
        self.weights = weights or {name: 1.0 for name in strategies}
        self.min_agreement = min_agreement
        self.weighting_mode = weighting_mode

    def evaluate(self, data: pd.DataFrame) -> Signal:
        """Run all sub-strategies and merge their signals."""
        sub_signals: dict[str, Signal] = {}

        for name, strategy in self.strategies.items():
            try:
                sig = strategy.evaluate(data)
                sub_signals[name] = sig
            except Exception as exc:
                logger.warning("Strategy %s failed: %s", name, exc)
                sub_signals[name] = Signal(
                    signal_type=SignalType.HOLD,
                    confidence=0.0,
                    strategy_name=name,
                )

        return self.merge_signals(sub_signals)

    def evaluate_with_signals(
        self,
        pre_computed: dict[str, Signal],
    ) -> Signal:
        """Merge pre-computed signals (useful when strategies need different data)."""
        return self.merge_signals(pre_computed)

    def merge_signals(self, signals: dict[str, Signal]) -> Signal:
        """Core merging logic.

        1. Compute weighted votes for BUY, SELL, HOLD.
        2. Pick the direction with the highest weighted vote.
        3. Check agreement threshold.
        4. Compute ensemble confidence.
        """
        if not signals:
            return Signal(signal_type=SignalType.HOLD, confidence=0.0, strategy_name="ensemble")

        # Accumulate weighted votes
        direction_scores: dict[SignalType, float] = {
            SignalType.BUY: 0.0,
            SignalType.SELL: 0.0,
            SignalType.HOLD: 0.0,
        }
        direction_counts: dict[SignalType, int] = {
            SignalType.BUY: 0,
            SignalType.SELL: 0,
            SignalType.HOLD: 0,
        }
        total_weight = 0.0
        weighted_confidence: dict[SignalType, float] = {
            SignalType.BUY: 0.0,
            SignalType.SELL: 0.0,
            SignalType.HOLD: 0.0,
        }

        for name, sig in signals.items():
            w = self._get_weight(name, sig)
            direction_scores[sig.signal_type] += w
            direction_counts[sig.signal_type] += 1
            weighted_confidence[sig.signal_type] += w * sig.confidence
            total_weight += w

        if total_weight == 0:
            return Signal(signal_type=SignalType.HOLD, confidence=0.0, strategy_name="ensemble")

        # Determine winner
        best_direction = max(direction_scores, key=lambda d: direction_scores[d])
        best_score = direction_scores[best_direction]
        agreement = best_score / total_weight

        meta: dict[str, Any] = {
            "sub_signals": {
                name: {
                    "type": sig.signal_type.value,
                    "confidence": sig.confidence,
                }
                for name, sig in signals.items()
            },
            "direction_scores": {d.value: s for d, s in direction_scores.items()},
            "agreement": agreement,
        }

        # Check agreement threshold
        if agreement < self.min_agreement:
            logger.info(
                "Ensemble: no consensus (agreement=%.2f < %.2f)",
                agreement,
                self.min_agreement,
            )
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                strategy_name="ensemble",
                metadata=meta,
            )

        # Compute ensemble confidence
        if direction_scores[best_direction] > 0:
            confidence = (
                weighted_confidence[best_direction]
                / direction_scores[best_direction]
            )
        else:
            confidence = 0.0

        # Scale confidence by agreement level
        confidence *= agreement

        return Signal(
            signal_type=best_direction,
            confidence=min(confidence, 1.0),
            strategy_name="ensemble",
            metadata=meta,
        )

    def _get_weight(self, name: str, signal: Signal) -> float:
        """Determine the weight for a strategy's signal."""
        base = self.weights.get(name, 1.0)
        if self.weighting_mode == "confidence":
            return base * (0.1 + signal.confidence)  # floor at 0.1 to not ignore low-conf
        return base
