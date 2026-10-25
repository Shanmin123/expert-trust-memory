"""Fundamental expert: statement-based fundamentals, filing text, risk state.

Expected context key: "fundamental" -> {"by_ticker": {ticker: {...}}, ...} where each
ticker block may carry:
  - "financials": PIT SEC-XBRL annual ratios (revenue_growth_yoy, gross/operating/net
    margin, roa, fcf_margin, leverage) from the latest 10-K public at t -- the PRIMARY,
    price-ORTHOGONAL evidence (offline corr vs technical +0.005 vs +0.308 for the old
    price-derived inputs). Wired via expert_context ``sec_facts_path`` ->
    ``alpha.sec_fundamentals.SecFundamentalsAlpha``.
  - "filing": Loughran-McDonald tone LEVEL + CHANGE (lm_*_delta) and cosine-TF-IDF
    disclosure change (Lazy Prices) from the latest 10-K/10-Q -- SECONDARY text evidence.
  - price-derived context (drawdown, realized_vol, breadth_above_50d) -- BACKGROUND only,
    since the technical expert already owns the price/trend lens.
All fields are optional; missing fields are handled gracefully. Macro inputs remain
stubbed off (macro_available False in expert_context).
The statement-based financials are what make this expert genuinely orthogonal to the
price-derived technical/factor experts.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from trading_agent.contracts import Observation
from trading_agent.experts.base_expert import BaseExpert, ExpertBackend, _BASE_SYSTEM_SUFFIX, _strip_for_prompt


class FundamentalExpert(BaseExpert):
    expert_id: str = "fundamental"

    def __init__(self, backend: ExpertBackend) -> None:
        super().__init__(backend)

    def _system_prompt(self) -> str:
        # Adapted from TradingAgents' fundamentals analyst (financial documents + basic
        # financials + financial history; external/research_refs/TradingAgents/.../
        # fundamentals_analyst.py), recast as this project's batch-JSON voter with the
        # SEC-statement ratios as the primary evidence.
        return (
            "You are a fundamentals researcher analyzing a company's financial documents "
            "and financial history for a long-only equity decision. "
            "Your PRIMARY evidence is per-ticker 'financials': ratios from the company's "
            "most recent ANNUAL (10-K) report already public at this date (SEC XBRL), "
            "which are orthogonal to price -- 'revenue_growth_yoy', 'gross_margin', "
            "'operating_margin', 'net_margin', 'roa', 'fcf_margin', and 'leverage' "
            "(liabilities/equity). A constructive fundamental picture is growing revenue, "
            "healthy or improving margins and ROA, positive free-cash-flow margin, and "
            "contained leverage; weak or deteriorating profitability with rising leverage "
            "supports flat. Some fields may be null (not all issuers report every line, "
            "e.g. banks have no gross margin) -- judge on what is present. "
            "A 'quarter' sub-block, when present, carries the latest 10-Q pulse -- "
            "'revenue_yoy_q' (same-quarter year-over-year growth) and 'net_margin_q' -- "
            "the freshest fundamental evidence between annual reports; weigh a clear "
            "quarterly deterioration or improvement against the annual picture. "
            "SECONDARY evidence is 'filing' text signals from the latest 10-K/10-Q. SEC "
            "filings read systematically negative in Loughran-McDonald terms, so do NOT "
            "treat a negative 'lm_negative'/'lm_polarity' LEVEL as bearish; judge the "
            "CHANGE vs the prior same-form filing -- 'lm_negative_delta'/'lm_polarity_delta' "
            "(worsening tone precedes weaker returns) and 'cosine_text_change' (large "
            "disclosure rewrites precede weaker returns, per the Lazy-Prices literature). "
            "The price-condition fields (drawdown, realized volatility, breadth_above_50d) "
            "are BACKGROUND only, since the technical expert already covers price/trend; do "
            "not base your judgment on them. Assess whether fundamentals support going long "
            "or staying flat. When the fundamental picture is genuinely mixed or immaterial, "
            "vote \"neutral\" (a deliberate abstention) and reserve \"flat\" for identifiable "
            "weakness; neutral is itself a judgment that must be supported by your evidence "
            "fields, not a lazy default. Do not use any market-state label."
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
        fundamental = _strip_for_prompt(context.get("fundamental", {}) if context else {})
        payload: dict[str, Any] = {
            "date": observation.decision_date.isoformat(),
            "n_tickers": len(observation.tickers),
            "tickers": list(observation.tickers),
            "fundamental_inputs": fundamental,
        }
        return json.dumps(payload, ensure_ascii=False)
