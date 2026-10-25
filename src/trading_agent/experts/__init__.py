"""The expert committee for the LLM trading agent.

Each expert receives a point-in-time Observation and a regime label, calls an
LLM backend with a focused system prompt, and returns an ExpertView. On any
parse failure the expert returns a safe default (direction="flat", confidence=0).
"""

from trading_agent.experts.base_expert import BaseExpert, ExpertBackend
from trading_agent.experts.factor_expert import FactorExpert
from trading_agent.experts.fundamental_expert import FundamentalExpert
from trading_agent.experts.news_expert import NewsExpert
from trading_agent.experts.technical_expert import TechnicalExpert
from trading_agent.experts.view_corpus import (
    CachedExpertBackend,
    CorpusRecord,
    ViewCorpus,
    attach_corpus,
)

__all__ = [
    "ExpertBackend",
    "BaseExpert",
    "TechnicalExpert",
    "FactorExpert",
    "NewsExpert",
    "FundamentalExpert",
    "ViewCorpus",
    "CachedExpertBackend",
    "CorpusRecord",
    "attach_corpus",
]
