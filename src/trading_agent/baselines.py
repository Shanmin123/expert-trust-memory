"""Deterministic, non-LLM baseline strategy for the smoke milestone.

Implements the AlphaEngine / RiskEngine / PortfolioEngine Protocols as an equal-weight
buy-and-hold: score every available ticker, target equal weight among positive signals,
and buy each target once (then hold). No randomness, no LLM — the point is to validate
the backtest engine, not alpha quality.
"""

from __future__ import annotations

from typing import Sequence

from trading_agent.contracts import (
    AccountState,
    Observation,
    OrderIntent,
    PortfolioState,
    Signal,
    TargetPosition,
)


class EqualWeightAlpha:
    """Emits a flat positive signal for every ticker present in the observation."""

    source_id = "equal_weight"

    def score(self, observation: Observation) -> Sequence[Signal]:
        return [
            Signal(
                ticker=ticker,
                value=1.0,
                source_id=self.source_id,
                as_of=observation.decision_date,
            )
            for ticker in observation.tickers
        ]


class EqualWeightRisk:
    """Equal weight across positive-value signals (long/flat)."""

    def size(
        self,
        signals: Sequence[Signal],
        account: AccountState,
        observation: Observation,
    ) -> Sequence[TargetPosition]:
        positive = [s for s in signals if s.value > 0]
        if not positive:
            return []
        weight = 1.0 / len(positive)
        return [
            TargetPosition(ticker=s.ticker, target_weight=weight, as_of=s.as_of)
            for s in positive
        ]


class BuyAndHoldPortfolio:
    """Buy each target once (when not yet held) toward its weight; otherwise hold."""

    def build_orders(
        self,
        targets: Sequence[TargetPosition],
        account: AccountState,
        positions: PortfolioState,
        observation: Observation,
    ) -> Sequence[OrderIntent]:
        held = positions.by_ticker()
        intents: list[OrderIntent] = []
        for target in targets:
            if target.ticker in held:  # already established -> hold
                continue
            bar = observation.bars.get(target.ticker)
            if bar is None or bar.adjusted_close <= 0:
                continue
            quantity = int((account.equity * target.target_weight) / bar.adjusted_close)
            if quantity > 0:
                intents.append(OrderIntent(ticker=target.ticker, side="buy", quantity=quantity))
        return intents


def equal_weight_engines():
    """Return the (alpha, risk, portfolio) triple for the equal-weight buy-and-hold baseline."""
    return EqualWeightAlpha(), EqualWeightRisk(), BuyAndHoldPortfolio()
