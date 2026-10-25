"""Technical expert. Design D: deterministic TradingView-style rating vote.

When the context carries a per-ticker ``rating`` block ({'summary','ma','oscillator'} in
[-1,1], built by expert_context from ``TradingViewRating``), the expert votes
DETERMINISTICALLY — long when summary > 0, flat otherwise — with no LLM call, mirroring the
public TradingView Technical Ratings rule set. Confidence carries |summary| (audit-only; the
aggregation reads direction).

Without a rating block (smoke tests, legacy runs) it falls back to the original LLM prompt
over time-series momentum/reversal/lowvol z-scores + the stockstats indicator block.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from trading_agent.contracts import ExpertView, Observation
from trading_agent.experts.base_expert import BaseExpert, ExpertBackend, _BASE_SYSTEM_SUFFIX, _strip_for_prompt


class TechnicalExpert(BaseExpert):
    expert_id: str = "technical"

    def __init__(self, backend: ExpertBackend) -> None:
        super().__init__(backend)

    @staticmethod
    def _rating_of(context: Optional[dict[str, Any]], ticker: str) -> Optional[dict]:
        if not context:
            return None
        block = context.get("technical", {})
        by_ticker = block.get("by_ticker", {}) if isinstance(block, dict) else {}
        row = by_ticker.get(ticker)
        rating = row.get("rating") if isinstance(row, dict) else None
        return rating if isinstance(rating, dict) and rating.get("summary") is not None else None

    def views(
        self,
        observation: Observation,
        regime_label: str,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, ExpertView]:
        ratings = {t: self._rating_of(context, t) for t in observation.tickers}
        if ratings and all(r is not None for r in ratings.values()):
            # Deterministic path (design D): every ticker has a rating -> no backend call.
            out: dict[str, ExpertView] = {}
            for ticker, r in ratings.items():
                summary = float(r["summary"])
                # TradingView's own verdict bands: Buy above +0.1, Neutral in
                # [-0.1, +0.1], Sell below -0.1. Neutral ABSTAINS from the vote (D3).
                if summary > 0.1:
                    direction, flags = "long", ()
                elif summary < -0.1:
                    direction, flags = "flat", ("rating_sell",)
                else:
                    direction, flags = "neutral", ("rating_neutral_band",)
                out[ticker] = ExpertView(
                    expert_id=self.expert_id,
                    direction=direction,
                    confidence=max(0.05, min(0.9, abs(summary))),
                    reasoning="deterministic TradingView technical rating (26 votes)",
                    evidence=(
                        f"summary={summary:.4f}",
                        f"ma={r.get('ma')}",
                        f"oscillator={r.get('oscillator')}",
                    ),
                    risk_flags=flags,
                    data_quality="high",
                    # Forced binary = the design-D sign rule (long when summary > 0),
                    # so the forced-binary replay reproduces the pre-D3 technical vote.
                    forced_direction="long" if summary > 0 else "flat",
                )
            return out
        return super().views(observation, regime_label, context)

    def _system_prompt(self) -> str:
        return (
            "You are a technical analysis expert for a long-only equity portfolio. "
            "Each summary signal compares the stock with its own trailing history as a "
            "time-series z-score: momentum (positive = trend stronger than the stock's "
            "own recent past), reversal (positive = more oversold than usual, bounce "
            "proposal), and lowvol (positive = calmer than usual, defensive). Where "
            "present, an 'indicators' block gives standard named readings: gap fields "
            "are close versus the 10-EMA/50-SMA/200-SMA (positive = price above the "
            "average), the MACD family is normalised by price, rsi uses the 70/30 "
            "overbought/oversold convention, and boll_position is +1 at the upper "
            "Bollinger band and -1 at the lower. Assess whether the technical picture "
            "supports going long or staying flat for each ticker. Weigh signal "
            "agreement, divergence, and volatility fit without using any market-state "
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
        tech = _strip_for_prompt(context.get("technical", {}) if context else {})
        payload: dict[str, Any] = {
            "date": observation.decision_date.isoformat(),
            "n_tickers": len(observation.tickers),
            "tickers": list(observation.tickers),
            "technical_signals": tech,
        }
        return json.dumps(payload, ensure_ascii=False)
