"""Expert backend implementations and factories."""

from __future__ import annotations

import json
import math
import importlib.util
from statistics import fmean
from typing import Any
from typing import Sequence

from trading_agent.contracts import Expert
from trading_agent.experts.fundamental_expert import FundamentalExpert
from trading_agent.experts.news_expert import NewsExpert
from trading_agent.experts.technical_expert import TechnicalExpert

DEFAULT_EXPERT_MODEL = "gpt-4.1-mini"
DEFAULT_CODEX_SDK_MODEL = "gpt-5.4-mini"
SUPPORTED_EXPERT_BACKENDS = ("static", "heuristic", "openai", "codex_sdk")


class StaticExpertBackend:
    """Deterministic backend for reproducible smoke and regression runs."""

    def __init__(self, confidence: float, *, reasoning: str = "static_reproducible") -> None:
        self._confidence = confidence
        self._reasoning = reasoning

    def call(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        try:
            tickers = json.loads(user_prompt).get("tickers", [])
        except Exception:
            tickers = []
        if not tickers:
            tickers = ["__portfolio__"]
        return json.dumps([
            {
                "ticker": ticker,
                "direction": "long",
                "confidence": self._confidence,
                "reasoning": self._reasoning,
                "evidence": ["deterministic reproducible expert view"],
                "risk_flags": [],
                "data_quality": "medium",
            }
            for ticker in tickers
        ])


class HeuristicExpertBackend:
    """Deterministic data-aware backend for real-data diagnostics without LLM calls."""

    def call(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        payload = json.loads(user_prompt)
        tickers = list(payload.get("tickers") or ["__portfolio__"])
        if "technical_signals" in payload:
            views = self._technical_views(tickers, payload.get("technical_signals") or {})
        elif "factor_signals" in payload:
            views = self._factor_views(tickers, payload.get("factor_signals") or {})
        elif "news_signals" in payload:
            views = self._news_views(tickers, payload.get("news_signals") or {})
        elif "fundamental_inputs" in payload:
            views = self._fundamental_views(tickers, payload.get("fundamental_inputs") or {})
        else:
            views = [self._view(ticker, "flat", 0.0, "no structured signal", [], ["no_signal"], "low")
                     for ticker in tickers]
        return json.dumps(views)

    def _technical_views(self, tickers: list[str], block: dict[str, Any]) -> list[dict[str, Any]]:
        by_ticker = _by_ticker(block)
        out = []
        for ticker in tickers:
            row = by_ticker.get(ticker, {})
            score = _mean_numeric(row, ("momentum", "reversal", "lowvol"))
            out.append(self._scored_view(
                ticker,
                score,
                source="technical heuristic",
                positive_evidence=f"technical_score={score:.4f}",
            ))
        return out

    def _factor_views(self, tickers: list[str], block: dict[str, Any]) -> list[dict[str, Any]]:
        by_ticker = _by_ticker(block)
        out = []
        for ticker in tickers:
            row = by_ticker.get(ticker, {})
            score = _mean_all_numeric(row, exclude={"available"})
            out.append(self._scored_view(
                ticker,
                score,
                source="factor heuristic",
                positive_evidence=f"factor_score={score:.4f}",
            ))
        return out

    def _news_views(self, tickers: list[str], block: dict[str, Any]) -> list[dict[str, Any]]:
        by_ticker = _by_ticker(block)
        out = []
        for ticker in tickers:
            row = by_ticker.get(ticker, {})
            sentiment = _optional_float(row.get("sentiment"))
            attention = _optional_float(row.get("attention_novelty"))
            raw_count = _optional_float(row.get("raw_item_count")) or 0.0
            components = []
            if sentiment is not None:
                components.append(sentiment)
            if attention is not None:
                components.append(0.5 * attention)
            score = fmean(components) if components else (0.05 if raw_count > 0 else -0.05)
            out.append(self._scored_view(
                ticker,
                score,
                source="news heuristic",
                positive_evidence=f"news_score={score:.4f}",
                data_quality=str(block.get("data_quality") or "medium"),
            ))
        return out

    def _fundamental_views(self, tickers: list[str], block: dict[str, Any]) -> list[dict[str, Any]]:
        by_ticker = _by_ticker(block)
        breadth = _optional_float(block.get("breadth_above_50d"))
        market_vol = _optional_float(block.get("realized_vol_20d"))
        mean_drawdown = _optional_float(block.get("mean_drawdown_60d"))
        market_score = 0.0
        if breadth is not None:
            market_score += (breadth - 0.5) * 2.0
        if market_vol is not None:
            market_score -= max(0.0, market_vol - 0.25)
        if mean_drawdown is not None:
            market_score += max(-1.0, mean_drawdown) * 0.5
        out = []
        for ticker in tickers:
            row = by_ticker.get(ticker, {})
            drawdown = _optional_float(row.get("drawdown_60d"))
            vol = _optional_float(row.get("realized_vol_20d"))
            ticker_score = market_score
            if drawdown is not None:
                ticker_score += max(-1.0, drawdown) * 0.5
            if vol is not None:
                ticker_score -= max(0.0, vol - 0.35) * 0.5
            out.append(self._scored_view(
                ticker,
                ticker_score,
                source="fundamental heuristic",
                positive_evidence=f"fundamental_score={ticker_score:.4f}",
            ))
        return out

    def _scored_view(
        self,
        ticker: str,
        score: float,
        *,
        source: str,
        positive_evidence: str,
        data_quality: str = "medium",
    ) -> dict[str, Any]:
        direction = "long" if score > 0.0 else "flat"
        confidence = _confidence_from_score(score)
        return self._view(
            ticker,
            direction,
            confidence,
            source,
            [positive_evidence],
            [] if direction == "long" else ["negative_or_weak_signal"],
            data_quality,
        )

    @staticmethod
    def _view(
        ticker: str,
        direction: str,
        confidence: float,
        reasoning: str,
        evidence: list[str],
        risk_flags: list[str],
        data_quality: str,
    ) -> dict[str, Any]:
        return {
            "ticker": ticker,
            "direction": direction,
            "confidence": round(float(confidence), 6),
            "reasoning": reasoning,
            "evidence": evidence or ["structured signal unavailable"],
            "risk_flags": risk_flags,
            "data_quality": data_quality if data_quality in {"high", "medium", "low", "unknown"} else "unknown",
        }


def _by_ticker(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = block.get("by_ticker", {}) if isinstance(block, dict) else {}
    return value if isinstance(value, dict) else {}


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
    # Non-finite values pass through DELIBERATELY: the frozen view corpora
    # were recorded with "nan" rendered into the prompts, and filtering here
    # changes the prompt bytes -> every cache key -> silently breaks replay
    # (found 2026-07-27 when a replay missed on news/fundamental for
    # every date). Guarding non-finite inputs belongs upstream in the data
    # pipeline, never between a frozen prompt and its corpus.


def _mean_numeric(row: dict[str, Any], keys: Sequence[str]) -> float:
    values = [_optional_float(row.get(key)) for key in keys]
    clean = [value for value in values if value is not None]
    return fmean(clean) if clean else -0.05


def _mean_all_numeric(row: dict[str, Any], *, exclude: set[str]) -> float:
    values = [
        _optional_float(value)
        for key, value in row.items()
        if key not in exclude and not isinstance(value, bool)
    ]
    clean = [value for value in values if value is not None]
    return fmean(clean) if clean else -0.05


def _confidence_from_score(score: float) -> float:
    return max(0.05, min(0.9, 0.20 + 0.35 * abs(float(score))))


class OpenAIChatBackend:
    """Optional OpenAI chat backend for real expert calls."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.0,
        model_revision: str | None = None,
    ) -> None:
        self.model = model
        self.model_name = model
        self.model_revision = model_revision or model
        self.temperature = float(temperature)

    def call(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for --expert-backend openai; "
                "install the optional LLM dependency and set OPENAI_API_KEY."
            ) from exc
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI expert backend returned an empty response")
        return content


class CodexSDKExpertBackend:
    """Optional Codex SDK backend for real expert calls via openai_codex."""

    transport = "openai_codex"
    temperature_policy = "deterministic_requested"

    def __init__(
        self,
        *,
        model: str = DEFAULT_CODEX_SDK_MODEL,
        temperature: float = 0.0,
        model_revision: str | None = None,
    ) -> None:
        self.model = model
        self.model_name = model
        self.model_revision = model_revision or model
        self.temperature = float(temperature)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai_codex import Codex
        except ImportError as exc:
            raise RuntimeError(
                "openai_codex package is required for --expert-backend codex_sdk; "
                "install/enable the Codex SDK and authenticate the account."
            ) from exc
        self._client = Codex()
        return self._client

    def close(self) -> None:
        client = self._client
        self._client = None
        close = getattr(client, "close", None)
        if callable(close):
            close()

    def call(self, system_prompt: str, user_prompt: str) -> str:
        prompt = (
            "System prompt:\n"
            f"{system_prompt}\n\n"
            "User prompt:\n"
            f"{user_prompt}\n\n"
            "Return only the JSON requested by the system prompt."
        )
        client = self._get_client()
        try:
            thread = client.thread_start(
                model=self.model,
                developer_instructions=system_prompt,
                ephemeral=True,
            )
            result = thread.run(prompt, model=self.model)
        except Exception as exc:
            raise RuntimeError(f"Codex SDK expert backend failed: {exc}") from exc

        content = _codex_result_text(result)
        if not content:
            raise RuntimeError("Codex SDK expert backend returned an empty response")
        return content


def _codex_result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    final_response = getattr(result, "final_response", None)
    if isinstance(final_response, str):
        return final_response
    if final_response is not None:
        return str(final_response)
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    return ""


def expert_backend_metadata(
    backend_name: str,
    *,
    model: str,
    model_revision: str | None = None,
    temperature: float = 0.0,
) -> dict[str, object]:
    name = backend_name.lower().strip()
    if name == "codex_sdk":
        return {
            "expert_transport": CodexSDKExpertBackend.transport,
            "expert_temperature": float(temperature),
            "expert_temperature_policy": CodexSDKExpertBackend.temperature_policy,
            "codex_sdk_available": importlib.util.find_spec("openai_codex") is not None,
        }
    if name == "openai":
        return {
            "expert_transport": "openai_api",
            "expert_temperature": float(temperature),
            "expert_temperature_policy": "temperature_zero",
            "codex_sdk_available": importlib.util.find_spec("openai_codex") is not None,
        }
    return {
        "expert_transport": name,
        "expert_temperature": float(temperature),
        "expert_temperature_policy": "deterministic",
        "codex_sdk_available": importlib.util.find_spec("openai_codex") is not None,
    }


def build_experts(
    backend_name: str = "static",
    *,
    model: str = DEFAULT_EXPERT_MODEL,
    temperature: float = 0.0,
    model_revision: str | None = None,
    participation_prior: bool = False,
) -> Sequence[Expert]:
    """Build the three fixed experts (design D) with the requested backend.

    ``participation_prior=True`` appends the upward-drift base rate to each
    expert's system prompt to counter the flat-bias (opt-in experiment).
    """
    name = backend_name.lower().strip()
    if name == "static":
        backends = [
            StaticExpertBackend(0.90, reasoning="static_technical"),
            StaticExpertBackend(0.50, reasoning="static_news"),
            StaticExpertBackend(0.45, reasoning="static_fundamental"),
        ]
    elif name == "heuristic":
        backends = [HeuristicExpertBackend() for _ in range(3)]
    elif name == "openai":
        backends = [
            OpenAIChatBackend(
                model=model,
                temperature=temperature,
                model_revision=model_revision,
            )
            for _ in range(3)
        ]
    elif name == "codex_sdk":
        backends = [
            CodexSDKExpertBackend(
                model=model or DEFAULT_CODEX_SDK_MODEL,
                temperature=temperature,
                model_revision=model_revision,
            )
            for _ in range(3)
        ]
    else:
        raise ValueError(f"unknown expert backend: {backend_name}")

    from trading_agent.experts.base_expert import PARTICIPATION_PRIOR

    experts = [
        TechnicalExpert(backends[0]),
        NewsExpert(backends[1]),
        FundamentalExpert(backends[2]),
    ]
    if participation_prior:
        for expert in experts:
            expert.participation_prior = PARTICIPATION_PRIOR
    return experts


class ForcedBinaryExpert:
    """Frozen forced-binary replay policy (design D3 comparison, 2026-07-17).

    Wraps an expert and maps every DELIBERATE neutral onto the forced binary
    judgment the SAME response carried (``forced_direction``), so one recorded
    corpus supports both the ternary system and a design-D-style binary replay
    — the round-2 review's design for identifying the effect of the ternary
    vote itself (everything else held fixed). Spoken views and safe defaults
    pass through untouched. A neutral WITHOUT a forced_direction (old corpus
    rows, dropped field) is NOT a binary judgment: it degrades to the
    parse_error safe default, which the per-view records count.

    Wrap LAST — after the view corpus is attached — so caching keys on the
    inner expert's raw prompts and the mapping happens at replay time.
    """

    def __init__(self, inner: Expert) -> None:
        self._inner = inner
        self.expert_id = inner.expert_id

    def views(self, observation, regime_label, context=None):
        import dataclasses

        from trading_agent.experts.base_expert import _safe_default, is_safe_default_view

        out = {}
        for ticker, view in self._inner.views(observation, regime_label, context).items():
            if view.direction == "neutral" and not is_safe_default_view(view):
                if view.forced_direction in ("long", "flat"):
                    out[ticker] = dataclasses.replace(view, direction=view.forced_direction)
                else:
                    out[ticker] = _safe_default(view.expert_id)
            else:
                out[ticker] = view
        return out
