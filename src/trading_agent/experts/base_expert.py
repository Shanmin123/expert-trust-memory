"""Base expert: prompt assembly, LLM call, JSON parse, safe fallback.

Subclasses override only ``expert_id``, ``_system_prompt``, and ``_user_prompt``.
All error handling and the safe-default fallback live here.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from trading_agent.contracts import ExpertView, Observation

logger = logging.getLogger(__name__)


@runtime_checkable
class ExpertBackend(Protocol):
    """Minimal interface for an LLM call used by experts."""

    def call(self, system_prompt: str, user_prompt: str) -> str: ...


# Canonical definitions live in contracts (aggregation depends on them too);
# re-exported here for backward compatibility.
from trading_agent.contracts import SAFE_DEFAULT_REASONS as _SAFE_DEFAULT_REASONS
from trading_agent.contracts import is_safe_default_view as _contracts_is_safe_default_view
_VALID_DIRECTIONS = frozenset({"long", "neutral", "flat"})
_VALID_DATA_QUALITY = frozenset({"high", "medium", "low", "unknown"})

# Opt-in upward-drift base rate that counters the experts' flat-bias. Grounded in
# the equity risk premium (textbook, NOT future data -> no look-ahead bias); an
# A/B on matched contexts moved 52/120 calls flat->long and 0 long->flat.
PARTICIPATION_PRIOR = (
    " BASE RATE: over multi-day horizons equities drift upward (the equity risk "
    "premium), so a positive next-period return is somewhat more likely than not. "
    "Treat 'flat' as a POSITIVE claim that identifiable near-term downside "
    "outweighs this drift, not a neutral default for weak or mixed evidence. Favour "
    "'long' when signals are ambiguous or mildly constructive; reserve 'flat' for "
    "clear risk-off evidence."
)

_BASE_SYSTEM_SUFFIX = (
    # Date anchoring adapted from TradingAgents' analyst frame ("Today's date is X;
    # treat it as 'now'"): the decision date rides in the user payload, so the model
    # must be told it is "now" and that the provided inputs are the whole world.
    ' The "date" field in the input is today: treat it as now, and judge strictly '
    'from the provided inputs as of that date. '
    'Return a source-grounded JSON array with exactly one object per ticker. '
    'Each object must include "ticker" and these fields: '
    '"direction" (one of "long", "neutral" or "flat"; "long" = evidence supports upside; '
    '"flat" = identifiable downside, vote against; "neutral" = a deliberate no-edge '
    'judgment that ABSTAINS from the committee vote — use it when the evidence is '
    'genuinely mixed or immaterial, not as a lazy default), '
    '"forced_direction" (one of "long" or "flat": the direction you would pick if '
    '"neutral" were NOT available; when your direction is already "long" or "flat", '
    'repeat it here), '
    '"confidence" (float 0.0-1.0, your conviction), '
    '"reasoning" (string, max 200 chars, human audit only). '
    '"evidence" (array of 1-3 short strings citing concrete facts FROM THE PROVIDED '
    'INPUTS — a quoted snippet phrase or an indicator name with its value), '
    '"risk_flags" (array of 0-3 short strings naming key risks or data limits), '
    '"data_quality" (one of "high", "medium", "low", "unknown"). '
    "Aggregation reads only direction and confidence; the other fields are "
    "for audit and later reflection context. "
    "Return only valid JSON with no markdown fences."
)


def _string_tuple(value: Any, *, max_items: int = 3, max_chars: int = 160) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_items = value
    else:
        return ()

    out: list[str] = []
    for item in raw_items:
        text = " ".join(str(item).split())[:max_chars]
        if text:
            out.append(text)
        if len(out) >= max_items:
            break
    return tuple(out)


def _safe_default(expert_id: str, reason: str = "parse_error") -> ExpertView:
    if reason not in _SAFE_DEFAULT_REASONS:
        reason = "parse_error"
    return ExpertView(
        expert_id=expert_id,
        direction="flat",
        confidence=0.0,
        reasoning=reason,
        evidence=(),
        risk_flags=(reason,),
        data_quality="low",
    )


def is_safe_default_view(view: ExpertView) -> bool:
    """True for parser/data-unavailable defaults that should not train reflection."""
    return _contracts_is_safe_default_view(view)


def safe_default_reason(view: ExpertView) -> str | None:
    if not is_safe_default_view(view):
        return None
    if view.reasoning in _SAFE_DEFAULT_REASONS:
        return view.reasoning
    return "parse_error"


# Audit-only telemetry that bloats the prompt with tokens the model cannot use
# (cache counters, config hashes, model revisions). Kept in the stored context for
# reproducibility; stripped from what the expert LLM actually reads.
_PROMPT_AUDIT_NOISE = {
    "nlp_cache_hits", "nlp_cache_misses", "nlp_config_hash",
    "nlp_model_revision", "nlp_item_count",
}


def _strip_for_prompt(value: Any) -> Any:
    """Remove fields that must never enter expert prompts.

    ``prior_lessons`` is deliberately NOT stripped (E-SKILL, 2026-07-18): it is
    the fed arm's input-side treatment channel. It only appears in context when
    ``--prior-lessons-path`` feeds it, so the unfed system's prompts stay
    byte-identical (pinned by test). market_state/regime stay stripped — the
    experts are regime-blind by design.
    """
    if isinstance(value, dict):
        return {
            key: _strip_for_prompt(val)
            for key, val in value.items()
            if key not in ({"market_state", "regime", "raw_text", "text"}
                           | _PROMPT_AUDIT_NOISE)
        }
    if isinstance(value, list):
        return [_strip_for_prompt(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_for_prompt(item) for item in value]
    return value


class BaseExpert:
    """Base class for the experts.

    Not abstract: usable directly in tests with a custom system prompt.
    """

    expert_id: str = "base"

    def __init__(self, backend: ExpertBackend) -> None:
        self._backend = backend
        self.last_raw_response: str | None = None
        # Opt-in participation prior appended to the concrete expert's system
        # prompt (before the shared suffix). Empty = the original neutral framing.
        self.participation_prior: str = ""

    def _system_prompt(self) -> str:
        return ("You are a trading expert assessing whether to go long or stay flat."
                + self.participation_prior + _BASE_SYSTEM_SUFFIX)

    def _user_prompt(
        self,
        observation: Observation,
        regime_label: str,
        context: Optional[dict[str, Any]],
    ) -> str:
        del regime_label
        payload: dict[str, Any] = {
            "date": observation.decision_date.isoformat(),
            "n_tickers": len(observation.tickers),
            "tickers": list(observation.tickers),
        }
        if context is not None:
            payload["signals"] = _strip_for_prompt(context)
        return json.dumps(payload, ensure_ascii=False)

    def _parse_data(self, data: Any) -> ExpertView:
        direction = str(data["direction"])
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(f"invalid direction: {direction!r}")
        confidence = float(data["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence out of range: {confidence}")
        reasoning = str(data.get("reasoning", ""))[:200]
        data_quality = str(data.get("data_quality", "unknown")).lower()
        if data_quality not in _VALID_DATA_QUALITY:
            data_quality = "unknown"
        # D3 forced-binary field: optional and LENIENT (a missing or invalid
        # value must never fail an otherwise valid judgment into a safe default).
        forced = data.get("forced_direction")
        forced_direction = forced if forced in ("long", "flat") else None
        return ExpertView(
            expert_id=self.expert_id,
            direction=direction,  # type: ignore[arg-type]
            confidence=confidence,
            reasoning=reasoning,
            evidence=_string_tuple(data.get("evidence")),
            risk_flags=_string_tuple(data.get("risk_flags")),
            data_quality=data_quality,  # type: ignore[arg-type]
            forced_direction=forced_direction,
        )

    def _parse(self, raw: str) -> ExpertView:
        data = json.loads(raw.strip(), strict=False)
        if isinstance(data, list):
            if not data:
                raise ValueError("empty expert view array")
            data = data[0]
        return self._parse_data(data)

    def _unavailable_tickers(
        self,
        observation: Observation,
        context: Optional[dict[str, Any]],
    ) -> set[str]:
        if context is None:
            return set()
        block = context.get(self.expert_id, {})
        by_ticker = block.get("by_ticker", {}) if isinstance(block, dict) else {}
        return {
            ticker for ticker in observation.tickers
            if isinstance(by_ticker.get(ticker), dict)
            and by_ticker[ticker].get("available") is False
        }

    def views(
        self,
        observation: Observation,
        regime_label: str,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, ExpertView]:
        tickers = list(observation.tickers)
        unavailable = self._unavailable_tickers(observation, context)
        if unavailable >= set(tickers):
            # Every ticker's input is marked unavailable: the answer is the
            # data_unavailable safe default regardless of what the backend
            # would say, so skip the call (saves one LLM round-trip per
            # no-data date, e.g. the news expert before news coverage starts).
            return {ticker: _safe_default(self.expert_id, "data_unavailable") for ticker in tickers}
        try:
            raw = self._backend.call(
                self._system_prompt(),
                self._user_prompt(observation, regime_label, context),
            )
            self.last_raw_response = raw
            data = json.loads(raw.strip(), strict=False)
            if not isinstance(data, list):
                data = [data]
        except Exception:
            logger.warning("expert %s batch response could not be parsed: %r", self.expert_id, self.last_raw_response)
            return {ticker: _safe_default(self.expert_id) for ticker in tickers}

        out: dict[str, ExpertView] = {ticker: _safe_default(self.expert_id) for ticker in tickers}
        seen: set[str] = set()
        for item in data:
            try:
                ticker = str(item["ticker"])
                if ticker not in out or ticker in seen:
                    continue
                seen.add(ticker)
                out[ticker] = self._parse_data(item)
            except Exception:
                if isinstance(item, dict) and str(item.get("ticker")) in out:
                    out[str(item.get("ticker"))] = _safe_default(self.expert_id)
        for ticker in unavailable:
            out[ticker] = _safe_default(self.expert_id, "data_unavailable")
        return out

    def view(
        self,
        observation: Observation,
        regime_label: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ExpertView:
        try:
            raw = self._backend.call(
                self._system_prompt(),
                self._user_prompt(observation, regime_label, context),
            )
            self.last_raw_response = raw
            return self._parse(raw)
        except Exception:
            return _safe_default(self.expert_id)
