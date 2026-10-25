"""News expert: FINSABER news attention/novelty and LLM sentiment signals.

Expected context key: "news"
  {
    "signals": {
        "news_attention_novelty": {...},
        "llm_news_sentiment": {...}
    },
    "nlp_model_name": str,
    "nlp_model_revision": str | None
  }
attention: abnormal news volume / novelty vs the name's own prior history
(positive = elevated attention).
sentiment: raw LLM news polarity in [-1, 1] (positive = bullish).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from trading_agent.contracts import Observation
from trading_agent.experts.base_expert import BaseExpert, ExpertBackend, _BASE_SYSTEM_SUFFIX, _strip_for_prompt


class NewsExpert(BaseExpert):
    expert_id: str = "news"

    def __init__(self, backend: ExpertBackend) -> None:
        super().__init__(backend)

    def _system_prompt(self) -> str:
        # Adapted from TradingAgents' news analyst (news + macroeconomics in ONE lens;
        # external/research_refs/TradingAgents/.../news_analyst.py), recast as this
        # project's single-shot batch-JSON voter.
        return (
            "You are a news researcher analyzing recent news and macroeconomic trends for "
            "a long-only equity decision. Per ticker you receive point-in-time inputs: "
            "'recent_news' (the latest news text snippets from the trailing week), and "
            "structured features (coverage, item counts, attention/novelty vs the stock's "
            "own history, LLM-derived sentiment polarity). A shared 'macro' block grounds "
            "the macroeconomic backdrop in FRED data: fed_funds_rate and "
            "fed_funds_change_3m (the hiking/cutting cycle), yield_curve_10y_minus_2y "
            "(inversion = recession signal), cpi_yoy_pct (inflation), initial_claims and "
            "unemployment_rate (labour). Weigh company-specific news against that "
            "backdrop: strong company news in a supportive macro regime is constructive; "
            "rapid hikes, curve inversion, or rising claims temper bullish company news, "
            "and macro shocks (wars, policy ruptures) usually surface as attention spikes "
            "with negative sentiment. Assess whether the news picture supports going long "
            "or staying flat for each ticker. When coverage is sparse or signals disagree, "
            "lower your confidence rather than defaulting to flat; when the picture is "
            "genuinely two-sided or immaterial, vote \"neutral\" (a deliberate abstention) "
            "and reserve \"flat\" for identifiable downside."
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
        news = _strip_for_prompt(context.get("news", {}) if context else {})
        payload: dict[str, Any] = {
            "date": observation.decision_date.isoformat(),
            "n_tickers": len(observation.tickers),
            "tickers": list(observation.tickers),
            "news_signals": news,
        }
        return json.dumps(payload, ensure_ascii=False)
