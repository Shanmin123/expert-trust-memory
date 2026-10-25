"""Factor expert: an LLM reads time-series-normalised Alpha158 features.

The LLM (not a ridge/ML ranker -- there is no such model in this path) receives the
registered Alpha158 features and outputs a direction. Each feature is a TIME-SERIES
z-score against the same ticker's own trailing year (positive = above the stock's own
recent norm). Raw Alpha158 ratios hover around 1.0 and would read as always-positive; a
cross-sectional z-score would degenerate to 0 for a single-name universe.

Expected context key: "factor" -> {"by_ticker": {ticker: {alpha158_*: z-score, ...}}}.

CAVEAT: Alpha158 is a CROSS-SECTIONAL stock-selection feature set;
used single-name and read by an LLM (which only weakly tracks the features), this expert
is the weakest-justified of the four. The financials/disclosure fundamental split is the
evidence-backed replacement -- see the final-design note.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from trading_agent.contracts import Observation
from trading_agent.experts.base_expert import BaseExpert, ExpertBackend, _BASE_SYSTEM_SUFFIX, _strip_for_prompt


class FactorExpert(BaseExpert):
    expert_id: str = "factor"

    def __init__(self, backend: ExpertBackend) -> None:
        super().__init__(backend)

    def _system_prompt(self) -> str:
        return (
            "You are a quantitative factor expert for a long-only equity portfolio. "
            "You receive Alpha158 factor readings (momentum, trend, volatility, "
            "range, volume families), each expressed as a time-series z-score of "
            "that feature against the same stock's own trailing year: positive = "
            "above the stock's own recent norm, negative = below it, near zero = "
            "typical. Assess whether the factor picture supports going long or "
            "staying flat for each ticker: broad positive momentum/trend readings "
            "with contained volatility readings support long; broadly negative or "
            "deteriorating readings support flat. Discuss agreement across factor "
            "families and possible model fragility without using any market-state "
            "label."
            + self.participation_prior
            + _BASE_SYSTEM_SUFFIX
        )

    def _user_prompt(
        self,
        observation: Observation,
        regime_label: str,
        context: Optional[dict[str, Any]],
    ) -> str:
        del regime_label
        factor = _strip_for_prompt(context.get("factor", {}) if context else {})
        payload: dict[str, Any] = {
            "date": observation.decision_date.isoformat(),
            "n_tickers": len(observation.tickers),
            "tickers": list(observation.tickers),
            "factor_signals": factor,
        }
        return json.dumps(payload, ensure_ascii=False)
