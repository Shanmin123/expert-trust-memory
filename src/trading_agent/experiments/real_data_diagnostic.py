from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Mapping, Sequence

from trading_agent.alpha.market_state_classifier import MarketStateConfig
from trading_agent.alpha.alpha158_strategy import available_tickers_from_sqlite_qlib_factors
from trading_agent.aggregation.soft_weighting import aggregate_by_ticker
from trading_agent.contracts import BacktestConfig
from trading_agent.data.finsaber_panel import load_panel, resolve_data_root, select_fixed_universe
from trading_agent.execution.finsaber_adapter import FinsaberBacktestAdapter
from trading_agent.expert_context import ExpertSignalContextBuilder, ExpertSignalSources, build_expert_signal_sources
from trading_agent.experts.base_expert import is_safe_default_view, safe_default_reason
from trading_agent.experts.backends import (
    DEFAULT_CODEX_SDK_MODEL,
    DEFAULT_EXPERT_MODEL,
    SUPPORTED_EXPERT_BACKENDS,
    OpenAIChatBackend,
    build_experts,
    expert_backend_metadata,
)
from trading_agent.market import LoaderPricePanel
from trading_agent.memory.expert_weights import MARKET_STATES


DEFAULT_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)
REAL_JUDGMENT_GATE = 0.8


@dataclass
class SignalOnlyDiagnosticResult:
    n_dates: int
    n_active_dates: int
    market_states: list[tuple]
    confidence_diagnostics: dict
    expert_view_records: dict
    equal_weight_score_records: list
    context_records: list
    episode_counts_by_state_action: dict
    parallel_serial_check: dict = field(default_factory=lambda: {
        "enabled": False,
        "passed": None,
        "comparisons": [],
    })


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_thresholds(value: str) -> tuple[float, ...]:
    out = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if raw:
            out.append(float(raw))
    if not out:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    for threshold in out:
        if not 0.0 <= threshold <= 1.0:
            raise argparse.ArgumentTypeError("thresholds must be in [0, 1]")
    return tuple(out)


def _parse_reserved(value: str) -> dict[str, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("reserved periods must be name:YYYY-MM-DD:YYYY-MM-DD")
    name, start, end = parts
    _parse_date(start)
    _parse_date(end)
    if _parse_date(start) > _parse_date(end):
        raise argparse.ArgumentTypeError("reserved period start must be on or before end")
    return {"name": name, "start": start, "end": end}


def _overlaps(start: date, end: date, period: Mapping[str, str]) -> bool:
    p_start = _parse_date(str(period["start"]))
    p_end = _parse_date(str(period["end"]))
    return start <= p_end and p_start <= end


def _require_no_reserved_overlap(start: date, end: date, periods: Sequence[Mapping[str, str]]) -> None:
    for period in periods:
        if _overlaps(start, end, period):
            raise ValueError(
                f"diagnostic window overlaps reserved test period {period['name']}: "
                f"{period['start']} to {period['end']}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a DIAGNOSTIC liveness pass on non-reserved data.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--start", type=_parse_date, required=True)
    parser.add_argument("--end", type=_parse_date, required=True)
    parser.add_argument("--names", type=int, default=10)
    parser.add_argument("--reserved-test-period", action="append", type=_parse_reserved, default=[])
    parser.add_argument("--decision-thresholds", type=_parse_thresholds, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--reflect-every", type=int, default=21)
    parser.add_argument("--reflection-window", type=int, default=63)
    parser.add_argument("--reflection-min-support", type=int, default=5)
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--headline-cost-bps-per-side", type=float, default=10.0)
    parser.add_argument("--preload-calendar-days", type=int, default=900)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--vol-window", type=int, default=21)
    parser.add_argument("--percentile-window", type=int, default=504)
    parser.add_argument("--alpha158-factors-db", default=None)
    parser.add_argument("--qlib-provider-uri", default=None)
    parser.add_argument("--no-news", action="store_true")
    parser.add_argument("--strict-news", action="store_true")
    parser.add_argument("--expert-backend", choices=SUPPORTED_EXPERT_BACKENDS, default="heuristic")
    parser.add_argument("--expert-model", default=None)
    parser.add_argument("--expert-model-revision", default=None)
    parser.add_argument("--news-sentiment-cache", default=None)
    parser.add_argument("--news-sentiment-model", default="gpt-4.1-mini")
    parser.add_argument("--progress", action="store_true", help="Print per-date and per-expert diagnostic progress.")
    parser.add_argument("--parallel-experts", dest="parallel_experts", action="store_true", default=None)
    parser.add_argument("--serial-experts", dest="parallel_experts", action="store_false")
    parser.add_argument("--signal-only", action="store_true", default=True)
    parser.add_argument("--window-role", default="diagnostic_or_validation_only")
    parser.add_argument("--parallel-serial-check-days", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", default="reports/real_data_diagnostic")
    return parser


def _preload_start(start: date, calendar_days: int) -> date:
    return start - timedelta(days=max(0, int(calendar_days)))


def _normalise_expert_model_identity(args: argparse.Namespace) -> None:
    if not getattr(args, "expert_model", None):
        args.expert_model = (
            DEFAULT_CODEX_SDK_MODEL
            if str(args.expert_backend).strip() == "codex_sdk"
            else DEFAULT_EXPERT_MODEL
        )
    if not getattr(args, "expert_model_revision", None):
        args.expert_model_revision = (
            str(args.expert_model)
            if str(args.expert_backend).strip() in {"openai", "codex_sdk"}
            else f"{args.expert_backend}-deterministic"
        )


def _factor_artifact_tickers(alpha158_factors_db: str | None) -> tuple[str, ...] | None:
    if not alpha158_factors_db:
        return None
    tickers = available_tickers_from_sqlite_qlib_factors(alpha158_factors_db)
    if not tickers:
        raise ValueError(f"Alpha158 factor artifact has no qlib_direct_* tables: {alpha158_factors_db}")
    return tickers


def _progress_printer(*, enabled: bool):
    def callback(event: Mapping[str, object]) -> None:
        if not enabled:
            return
        kind = str(event.get("event", ""))
        day = event.get("date", "")
        if kind == "date_start":
            print(f"[diagnostic] date_start {day}", flush=True)
        elif kind == "expert_start":
            print(f"[diagnostic] expert_start {day} {event.get('expert_id', '')}", flush=True)
        elif kind == "expert_done":
            print(
                f"[diagnostic] expert_done {day} {event.get('expert_id', '')} "
                f"n_tickers={event.get('n_tickers', 0)}",
                flush=True,
            )
        elif kind == "date_done":
            print(
                f"[diagnostic] date_done {day} state={event.get('market_state', '')} "
                f"active={event.get('active', False)} "
                f"long={event.get('long_count', 0)}/{event.get('n_tickers', 0)}",
                flush=True,
            )

    return callback


def _ten_bin_histogram(values: Sequence[float]) -> dict[str, int]:
    bins = {f"{i / 10:.1f}-{(i + 1) / 10:.1f}": 0 for i in range(10)}
    for raw in values:
        value = max(0.0, min(1.0, float(raw)))
        idx = min(9, int(value * 10))
        bins[f"{idx / 10:.1f}-{(idx + 1) / 10:.1f}"] += 1
    return bins


def _direction_counts(records: Sequence[Mapping[str, object]]) -> dict[str, float]:
    total = len(records)
    long_count = sum(1 for row in records if row.get("direction") == "long")
    flat_count = sum(1 for row in records if row.get("direction") == "flat")
    return {
        "n": total,
        "long_count": long_count,
        "flat_count": flat_count,
        "long_fraction": (long_count / total) if total else 0.0,
        "flat_fraction": (flat_count / total) if total else 0.0,
    }


def _confidence_distributions(records_by_expert: Mapping[str, Sequence[Mapping[str, object]]]) -> dict:
    out = {}
    for expert_id, records in sorted(records_by_expert.items()):
        all_values = [float(row.get("confidence", 0.0)) for row in records]
        long_values = [
            float(row.get("confidence", 0.0))
            for row in records
            if row.get("direction") == "long"
        ]
        out[expert_id] = {
            "all_histogram": _ten_bin_histogram(all_values),
            "long_histogram": _ten_bin_histogram(long_values),
        }
    return out


def _direction_mix(records_by_expert: Mapping[str, Sequence[Mapping[str, object]]]) -> dict:
    out = {}
    for expert_id, records in sorted(records_by_expert.items()):
        by_state: dict[str, list] = {}
        for row in records:
            by_state.setdefault(str(row.get("market_state", "unknown")), []).append(row)
        out[expert_id] = {
            "overall": _direction_counts(records),
            "by_state": {
                state: _direction_counts(rows)
                for state, rows in sorted(by_state.items())
            },
        }
    return out


def _expert_liveness(confidence_diagnostics: Mapping[str, Mapping[str, object]]) -> dict:
    return {
        expert_id: {
            "real_judgment_rate": float(diag.get("real_judgment_rate", 0.0)),
            "real_judgment_count": int(diag.get("real_judgment_count", 0)),
            "safe_default_count": int(diag.get("safe_default_count", 0)),
            "safe_default_reasons": _reason_counts(diag.get("safe_default_reasons", {})),
            "pass": float(diag.get("real_judgment_rate", 0.0)) >= REAL_JUDGMENT_GATE,
        }
        for expert_id, diag in sorted(confidence_diagnostics.items())
    }


def _reason_counts(raw: object = None) -> dict[str, int]:
    base = {"data_unavailable": 0, "parse_error": 0, "timeout": 0, "unknown": 0}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            reason = str(key)
            base[reason if reason in base else "unknown"] += int(value or 0)
    return base


def _score_quantiles(scores: Sequence[float]) -> dict[str, float | None]:
    if not scores:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    values = sorted(float(score) for score in scores)

    def percentile(pct: float) -> float:
        if len(values) == 1:
            return round(values[0], 6)
        pos = (len(values) - 1) * pct
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        frac = pos - lo
        return round(values[lo] * (1.0 - frac) + values[hi] * frac, 6)

    return {
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
    }


def _score_margin(score_records: Sequence[Mapping[str, object]], thresholds: Sequence[float]) -> dict:
    scores = [float(row.get("score", 0.0)) for row in score_records]
    thresholds_out = {}
    for threshold in thresholds:
        near = sum(1 for score in scores if abs(score - float(threshold)) <= 0.10)
        thresholds_out[f"{float(threshold):.4f}"] = {
            "within_0.10_count": near,
            "within_0.10_fraction": (near / len(scores)) if scores else 0.0,
        }
    return {
        "n": len(scores),
        "score_histogram": _ten_bin_histogram(scores),
        "quantiles": _score_quantiles(scores),
        "thresholds": thresholds_out,
    }


def _market_state_coverage(loop_result) -> dict:
    days: dict[str, set[str]] = {}
    for dt, state in getattr(loop_result, "market_states", []):
        days.setdefault(str(state), set()).add(str(dt))
    episodes: dict[str, dict[str, int]] = {}
    if hasattr(loop_result, "episode_counts_by_state_action"):
        episodes = {
            str(state): {str(action): int(count) for action, count in actions.items()}
            for state, actions in getattr(loop_result, "episode_counts_by_state_action", {}).items()
        }
    else:
        for episode in getattr(loop_result, "episodes", []):
            state = str(episode.market_state)
            action = str(getattr(episode, "action", "unknown"))
            episodes.setdefault(state, {"long": 0, "flat": 0})
            episodes[state][action] = episodes[state].get(action, 0) + 1
    active_states = sorted(state for state in episodes if state != "warmup")
    return {
        "states_seen": len(active_states),
        "days_per_state": {state: len(values) for state, values in sorted(days.items())},
        "episode_counts_by_state_action": episodes,
    }


def _expert_state_evidence(
    records_by_expert: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    min_support: int,
) -> dict[str, dict[str, dict[str, object]]]:
    states = sorted({
        str(row.get("market_state", "unknown"))
        for rows in records_by_expert.values()
        for row in rows
    })
    out: dict[str, dict[str, dict[str, object]]] = {}
    for state in states:
        out[state] = {}
        for expert_id, rows in sorted(records_by_expert.items()):
            scoped = [row for row in rows if str(row.get("market_state", "unknown")) == state]
            safe = [row for row in scoped if bool(row.get("safe_default"))]
            reasons = _reason_counts()
            for row in safe:
                reason = str(row.get("safe_default_reason") or "unknown")
                reasons[reason if reason in reasons else "unknown"] += 1
            non_default = len(scoped) - len(safe)
            out[state][expert_id] = {
                "non_default_evidence_count": non_default,
                "safe_default_count": len(safe),
                "safe_default_reasons": reasons,
                "passes_min_support": non_default >= int(min_support),
            }
    return out


def build_window_preflight(
    *,
    market_states: Sequence[tuple],
    tickers: Sequence[str],
    min_support: int,
    start: date,
    end: date,
    reserved_periods: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    _require_no_reserved_overlap(start, end, reserved_periods)
    days: dict[str, set[str]] = {}
    for day, state in market_states:
        state = str(state)
        if state in MARKET_STATES:
            days.setdefault(state, set()).add(str(day))
    state_support = {
        state: {
            "days": len(values),
            "estimated_per_name_observations": len(values) * len(tickers),
            "passes_min_support": len(values) * len(tickers) >= int(min_support),
        }
        for state, values in sorted(days.items())
    }
    supported = sum(1 for row in state_support.values() if row["passes_min_support"])
    window_pass = len(state_support) >= 3 and supported >= 3
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "window_role": "diagnostic_or_validation_only",
        "reserved_overlap": False,
        "states_seen": len(state_support),
        "supported_states": supported,
        "min_support": int(min_support),
        "n_tickers": len(tickers),
        "state_support": state_support,
        "window_pass": window_pass,
        "recommendation": "ok" if window_pass else "expand_window_do_not_lower_min_support",
    }


def _technical_unavailable(context_records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {"insufficient_trailing_history": 0, "signal_adapter_error": 0, "unknown": 0}
    for context in context_records:
        technical = context.get("technical", {}) if isinstance(context, Mapping) else {}
        by_ticker = technical.get("by_ticker", {}) if isinstance(technical, Mapping) else {}
        for row in by_ticker.values():
            if not isinstance(row, Mapping) or row.get("available") is not False:
                continue
            reason = str(row.get("unavailable_reason", "unknown"))
            counts[reason if reason in counts else "unknown"] += 1
    return counts


def _news_health(context_records: Sequence[Mapping[str, object]]) -> dict:
    raw_count = 0
    dedup_count = 0
    covered = 0
    total = 0
    model_name = None
    model_revision = None
    config_hash = None
    cache_hits = 0
    cache_misses = 0
    sentiment_items = 0
    for context in context_records:
        news = context.get("news", {}) if isinstance(context, Mapping) else {}
        if not isinstance(news, Mapping):
            continue
        raw_count += int(news.get("raw_item_count", 0) or 0)
        dedup_count += int(news.get("dedup_item_count", 0) or 0)
        model_name = model_name or news.get("nlp_model_name")
        model_revision = model_revision or news.get("nlp_model_revision")
        config_hash = config_hash or news.get("nlp_config_hash")
        cache_hits += int(news.get("nlp_cache_hits", 0) or 0)
        cache_misses += int(news.get("nlp_cache_misses", 0) or 0)
        sentiment_items += int(news.get("nlp_item_count", 0) or 0)
        by_ticker = news.get("by_ticker", {})
        if isinstance(by_ticker, Mapping):
            total += len(by_ticker)
            covered += sum(1 for row in by_ticker.values() if isinstance(row, Mapping) and row.get("available"))
    return {
        "raw_item_count": raw_count,
        "dedup_item_count": dedup_count,
        "nlp_model_name": model_name,
        "nlp_model_revision": model_revision,
        "nlp_config_hash": config_hash,
        "nlp_item_count": sentiment_items,
        "nlp_cache_hits": cache_hits,
        "nlp_cache_misses": cache_misses,
        "ticker_day_count": total,
        "covered_ticker_day_count": covered,
        "ticker_day_coverage_fraction": (covered / total) if total else 0.0,
    }


def _confidence_histogram(values: Sequence[float]) -> dict[str, int]:
    bins = {
        "0.0-0.2": 0,
        "0.2-0.4": 0,
        "0.4-0.6": 0,
        "0.6-0.8": 0,
        "0.8-1.0": 0,
    }
    for value in values:
        v = max(0.0, min(1.0, float(value)))
        if v < 0.2:
            bins["0.0-0.2"] += 1
        elif v < 0.4:
            bins["0.2-0.4"] += 1
        elif v < 0.6:
            bins["0.4-0.6"] += 1
        elif v < 0.8:
            bins["0.6-0.8"] += 1
        else:
            bins["0.8-1.0"] += 1
    return bins


def _confidence_summary(records: dict[str, list[dict]]) -> dict[str, dict]:
    out = {}
    for expert_id, rows in sorted(records.items()):
        values = [float(row["confidence"]) for row in rows]
        safe_rows = [row for row in rows if row["safe_default"]]
        reason_counts = _reason_counts()
        for row in safe_rows:
            reason = row.get("safe_default_reason") or "unknown"
            reason_counts[str(reason) if str(reason) in reason_counts else "unknown"] += 1
        mean = sum(values) / len(values) if values else 0.0
        real_count = len(rows) - len(safe_rows)
        out[expert_id] = {
            "n": len(rows),
            "mean": round(mean, 6),
            "min": round(min(values), 6) if values else 0.0,
            "max": round(max(values), 6) if values else 0.0,
            "histogram": _confidence_histogram(values),
            "long_count": sum(1 for row in rows if row["direction"] == "long"),
            "flat_count": sum(1 for row in rows if row["direction"] == "flat"),
            "safe_default_count": len(safe_rows),
            "safe_default_reasons": reason_counts,
            "real_judgment_count": real_count,
            "real_judgment_rate": (real_count / len(rows)) if rows else 0.0,
        }
    return out


def _view_signature(direction: str, confidence: float, safe_default: bool, reason: str | None) -> dict[str, object]:
    return {
        "direction": str(direction),
        "confidence": round(float(confidence), 6),
        "safe_default": bool(safe_default),
        "safe_default_reason": reason,
    }


def _signatures_from_views(views_by_expert: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    out: dict[str, dict[str, dict[str, object]]] = {}
    for expert_id, by_ticker in views_by_expert.items():
        out[expert_id] = {}
        for ticker, view in by_ticker.items():
            out[expert_id][ticker] = _view_signature(
                view.direction,
                float(view.confidence),
                is_safe_default_view(view),
                safe_default_reason(view),
            )
    return out


def compare_parallel_serial_views(comparisons: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Compare serial vs parallel expert-view signatures per date.

    Structural invariants (expert/ticker coverage, safe-default flag and reason)
    must match exactly and block the check when they differ. Direction and
    confidence are LLM judgments that may legitimately vary between runs, so
    their mismatches are audited (counted and surfaced) without failing the
    check.
    """
    rows = []
    passed = True
    direction_confidence_matched = True
    for item in comparisons:
        serial = item.get("serial", {}) or {}
        parallel = item.get("parallel", {}) or {}
        serial = serial if isinstance(serial, Mapping) else {}
        parallel = parallel if isinstance(parallel, Mapping) else {}
        ticker_coverage_mismatches = 0
        safe_default_mismatches = 0
        dc_mismatches = 0
        for expert_id in sorted(set(serial) | set(parallel)):
            serial_by_ticker = serial.get(expert_id) or {}
            parallel_by_ticker = parallel.get(expert_id) or {}
            for ticker in sorted(set(serial_by_ticker) | set(parallel_by_ticker)):
                serial_view = serial_by_ticker.get(ticker)
                parallel_view = parallel_by_ticker.get(ticker)
                if serial_view is None or parallel_view is None:
                    ticker_coverage_mismatches += 1
                    continue
                serial_structural = (
                    serial_view.get("safe_default"),
                    serial_view.get("safe_default_reason"),
                )
                parallel_structural = (
                    parallel_view.get("safe_default"),
                    parallel_view.get("safe_default_reason"),
                )
                if serial_structural != parallel_structural:
                    safe_default_mismatches += 1
                serial_judgment = (serial_view.get("direction"), serial_view.get("confidence"))
                parallel_judgment = (parallel_view.get("direction"), parallel_view.get("confidence"))
                if serial_judgment != parallel_judgment:
                    dc_mismatches += 1
        structural_matched = ticker_coverage_mismatches == 0 and safe_default_mismatches == 0
        passed = passed and structural_matched
        direction_confidence_matched = direction_confidence_matched and dc_mismatches == 0
        rows.append({
            "date": str(item.get("date")),
            "matched": structural_matched and dc_mismatches == 0,
            "structural_matched": structural_matched,
            "ticker_coverage_mismatches": ticker_coverage_mismatches,
            "safe_default_mismatches": safe_default_mismatches,
            "direction_confidence_mismatches": dc_mismatches,
            "serial_experts": sorted(serial.keys()),
            "parallel_experts": sorted(parallel.keys()),
        })
    return {
        "enabled": True,
        "passed": passed,
        "direction_confidence_matched": direction_confidence_matched,
        "comparisons": rows,
    }


def _call_experts(
    experts,
    observation,
    market_state: str,
    expert_context: Mapping[str, object],
    *,
    parallel: bool,
    progress_callback=None,
):
    def emit(event: str, **payload) -> None:
        if progress_callback is not None:
            progress_callback({"event": event, **payload})

    def call_one(expert):
        emit("expert_start", date=observation.decision_date, market_state=market_state, expert_id=expert.expert_id)
        views = expert.views(observation, market_state, context=expert_context)
        emit(
            "expert_done",
            date=observation.decision_date,
            market_state=market_state,
            expert_id=expert.expert_id,
            n_tickers=len(views),
        )
        return expert.expert_id, views

    out = {}
    if parallel:
        with ThreadPoolExecutor(max_workers=len(experts)) as pool:
            for future in as_completed([pool.submit(call_one, expert) for expert in experts]):
                expert_id, views = future.result()
                out[expert_id] = views
    else:
        for expert in experts:
            expert_id, views = call_one(expert)
            out[expert_id] = views
    return out


def run_signal_only_diagnostic(
    loader,
    config: BacktestConfig,
    experts,
    *,
    market_state_config=None,
    signal_sources: ExpertSignalSources | None = None,
    decision_threshold: float = 0.5,
    progress_callback=None,
    parallel_expert_calls: bool = False,
    parallel_serial_check_days: int = 0,
) -> SignalOnlyDiagnosticResult:
    from trading_agent.alpha.market_state_classifier import MarketStateClassifier

    adapter = FinsaberBacktestAdapter(loader, config)
    panel = LoaderPricePanel(adapter._window)
    classifier = MarketStateClassifier(panel, market_state_config or MarketStateConfig())
    context_builder = ExpertSignalContextBuilder(
        panel=panel,
        store=None,
        sources=signal_sources or ExpertSignalSources(),
    )
    market_states = []
    context_records = []
    confidence_records: dict[str, list[dict]] = {}
    equal_weight_score_records: list[dict] = []
    episode_counts: dict[str, dict[str, int]] = {}
    comparisons = []
    n_active_dates = 0

    for decision_date in adapter.dates:
        if progress_callback is not None:
            progress_callback({"event": "date_start", "date": decision_date})
        adapter.open_bar(decision_date)
        observation = adapter.get_observation(decision_date)
        market_state = classifier.classify(observation)
        market_states.append((decision_date, market_state))
        if market_state not in MARKET_STATES:
            adapter.advance()
            if progress_callback is not None:
                progress_callback({"event": "date_done", "date": decision_date, "market_state": market_state, "active": False})
            continue

        n_active_dates += 1
        expert_context = context_builder.build(observation, market_state)
        context_records.append(expert_context)

        if len(comparisons) < int(parallel_serial_check_days):
            serial_views = _call_experts(
                experts, observation, market_state, expert_context,
                parallel=False, progress_callback=None,
            )
            parallel_views = _call_experts(
                experts, observation, market_state, expert_context,
                parallel=True, progress_callback=None,
            )
            comparisons.append({
                "date": decision_date.isoformat(),
                "serial": _signatures_from_views(serial_views),
                "parallel": _signatures_from_views(parallel_views),
            })
            views_by_expert = parallel_views if parallel_expert_calls else serial_views
        else:
            views_by_expert = _call_experts(
                experts, observation, market_state, expert_context,
                parallel=parallel_expert_calls,
                progress_callback=progress_callback,
            )

        for expert_id, by_ticker in views_by_expert.items():
            rows = confidence_records.setdefault(expert_id, [])
            for ticker, view in by_ticker.items():
                rows.append({
                    "date": decision_date,
                    "ticker": ticker,
                    "market_state": market_state,
                    "confidence": float(view.confidence),
                    "direction": view.direction,
                    "safe_default": is_safe_default_view(view),
                    "safe_default_reason": safe_default_reason(view),
                })

        equal_agg = aggregate_by_ticker(
            views_by_expert,
            {},
            tickers=tuple(observation.tickers),
            threshold=float(decision_threshold),
        )
        for ticker, decision in equal_agg.by_ticker.items():
            equal_weight_score_records.append({
                "date": decision_date,
                "ticker": ticker,
                "market_state": market_state,
                "score": float(decision.score),
                "direction": decision.direction,
            })
            episode_counts.setdefault(market_state, {"long": 0, "flat": 0})
            episode_counts[market_state][decision.direction] = (
                episode_counts[market_state].get(decision.direction, 0) + 1
            )
        if progress_callback is not None:
            long_count = sum(1 for row in equal_agg.by_ticker.values() if row.direction == "long")
            progress_callback({
                "event": "date_done",
                "date": decision_date,
                "market_state": market_state,
                "active": True,
                "long_count": long_count,
                "n_tickers": len(observation.tickers),
            })
        adapter.advance()

    for expert in experts:
        close = getattr(getattr(expert, "_backend", None), "close", None)
        if callable(close):
            close()

    return SignalOnlyDiagnosticResult(
        n_dates=len(adapter.dates),
        n_active_dates=n_active_dates,
        market_states=market_states,
        confidence_diagnostics=_confidence_summary(confidence_records),
        expert_view_records=confidence_records,
        equal_weight_score_records=equal_weight_score_records,
        context_records=context_records,
        episode_counts_by_state_action=episode_counts,
        parallel_serial_check=(
            compare_parallel_serial_views(comparisons)
            if comparisons
            else {"enabled": False, "passed": None, "comparisons": []}
        ),
    )


def classify_market_states_only(
    loader,
    config: BacktestConfig,
    *,
    market_state_config=None,
) -> list[tuple]:
    from trading_agent.alpha.market_state_classifier import MarketStateClassifier

    adapter = FinsaberBacktestAdapter(loader, config)
    panel = LoaderPricePanel(adapter._window)
    classifier = MarketStateClassifier(panel, market_state_config or MarketStateConfig())
    states = []
    for decision_date in adapter.dates:
        adapter.open_bar(decision_date)
        observation = adapter.get_observation(decision_date)
        states.append((decision_date, classifier.classify(observation)))
        adapter.advance()
    return states


def write_preflight_report(payload: Mapping[str, object], output_base: str | Path) -> tuple[Path, Path]:
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    preflight = payload["window_preflight"]
    lines = [
        "# DIAGNOSTIC WINDOW PREFLIGHT",
        "",
        f"- window: {preflight['start']} to {preflight['end']}",
        f"- window_role: {preflight['window_role']}",
        f"- window_pass: {preflight['window_pass']}",
        f"- recommendation: {preflight['recommendation']}",
        f"- states_seen: {preflight['states_seen']}",
        f"- supported_states: {preflight['supported_states']}",
        "",
        "## State Support",
        "| state | days | estimated_per_name_observations | passes_min_support |",
        "|---|---:|---:|---:|",
    ]
    for state, row in preflight["state_support"].items():
        lines.append(
            f"| {state} | {row['days']} | {row['estimated_per_name_observations']} | "
            f"{row['passes_min_support']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _pass_gates(payload: Mapping[str, object]) -> dict[str, object]:
    expert_liveness = payload["expert_liveness"]
    all_live = all(row["pass"] for row in expert_liveness.values())
    coverage = payload["market_state_coverage"]
    score_margin = payload["score_margin"]
    min_support = int(payload["manifest"].get("reflection_min_support", payload["manifest"].get("min_support", 5)))
    expert_state_evidence = payload.get("expert_state_evidence", {})
    evidence_supported_states = sum(
        1
        for _state, experts in expert_state_evidence.items()
        if experts and all(row.get("passes_min_support") for row in experts.values())
    )
    margin_ok = any(
        row["within_0.10_count"] > 0
        for row in score_margin["thresholds"].values()
    )
    factor_reasons = expert_liveness.get("factor", {}).get("safe_default_reasons", {})
    factor_data_available = int(factor_reasons.get("data_unavailable", 0) or 0) == 0
    parallel_serial = payload.get("parallel_serial_check", {})
    parallel_serial_ok = (
        not parallel_serial.get("enabled")
        or bool(parallel_serial.get("passed"))
    )
    return {
        "all_experts_live": all_live,
        "market_state_coverage": coverage["states_seen"] >= 3,
        "market_state_min_support": evidence_supported_states >= 3,
        "expert_state_min_support": evidence_supported_states >= 3,
        "evidence_supported_states": evidence_supported_states,
        "factor_data_available": factor_data_available,
        "score_margin": margin_ok,
        "parallel_serial_check": parallel_serial_ok,
        "overall": bool(
            all_live
            and coverage["states_seen"] >= 3
            and evidence_supported_states >= 3
            and factor_data_available
            and margin_ok
            and parallel_serial_ok
        ),
    }


def build_diagnostic_payload(loop_result, *, manifest: Mapping[str, object]) -> dict:
    min_support = int(manifest.get("reflection_min_support", manifest.get("min_support", 5)))
    payload = {
        "mode": "DIAGNOSTIC",
        "manifest": dict(manifest),
        "expert_liveness": _expert_liveness(loop_result.confidence_diagnostics),
        "confidence_distributions": _confidence_distributions(loop_result.expert_view_records),
        "direction_mix": _direction_mix(loop_result.expert_view_records),
        "score_margin": _score_margin(
            loop_result.equal_weight_score_records,
            manifest.get("decision_thresholds", DEFAULT_THRESHOLDS),
        ),
        "market_state_coverage": _market_state_coverage(loop_result),
        "expert_state_evidence": _expert_state_evidence(
            loop_result.expert_view_records,
            min_support=min_support,
        ),
        "technical_unavailable": _technical_unavailable(loop_result.context_records),
        "news_health": _news_health(loop_result.context_records),
        "parallel_serial_check": getattr(loop_result, "parallel_serial_check", {
            "enabled": False,
            "passed": None,
            "comparisons": [],
        }),
        "window_preflight": dict(manifest.get("window_preflight", {})),
    }
    fallback_only = []
    for state, experts in payload["expert_state_evidence"].items():
        if not all(row["passes_min_support"] for row in experts.values()):
            fallback_only.append(state)
    payload["fallback_only_states"] = sorted(fallback_only)
    payload["pass_gates"] = _pass_gates(payload)
    news = payload["news_health"]
    if int(news.get("nlp_item_count", 0) or 0) > 0:
        missing = [
            key for key in ("nlp_model_name", "nlp_model_revision", "nlp_config_hash")
            if not news.get(key)
        ]
        if missing:
            raise ValueError(f"news sentiment metadata missing: {', '.join(missing)}")
    return payload


def payload_to_markdown(payload: Mapping[str, object]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# DIAGNOSTIC / SIGNAL-ONLY / NO PERFORMANCE CONCLUSIONS",
        "",
        "## Manifest",
        f"- diagnostic_window: {manifest['diagnostic_window']['start']} to {manifest['diagnostic_window']['end']}",
        f"- window_role: {manifest.get('window_role')}",
        f"- signal_only: {manifest.get('signal_only')}",
        f"- performance_metrics_computed: {manifest.get('performance_metrics_computed')}",
        f"- reserved_test_periods: {json.dumps(manifest['reserved_test_periods'])}",
        f"- expert_backend: {manifest['expert_backend']}",
        f"- expert_model: {manifest['expert_model']}",
        f"- expert_model_revision: {manifest['expert_model_revision']}",
        f"- expert_transport: {manifest['expert_transport']}",
        f"- expert_temperature: {manifest['expert_temperature']}",
        f"- parallel_expert_calls: {manifest.get('parallel_expert_calls')}",
        f"- news_sentiment_enabled: {manifest.get('news_sentiment_enabled')}",
        "",
        "## Expert Liveness",
        "| expert | real_rate | real | safe_default | data_unavailable | parse_error | pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for expert_id, row in payload["expert_liveness"].items():
        reasons = row["safe_default_reasons"]
        lines.append(
            f"| {expert_id} | {row['real_judgment_rate']:.4f} | {row['real_judgment_count']} | "
            f"{row['safe_default_count']} | {reasons.get('data_unavailable', 0)} | "
            f"{reasons.get('parse_error', 0)} | {row['pass']} |"
        )
    lines.extend([
        "",
        "## Confidence Distributions",
        "| expert | all_histogram | long_histogram |",
        "|---|---|---|",
    ])
    for expert_id, row in payload["confidence_distributions"].items():
        lines.append(
            f"| {expert_id} | `{json.dumps(row['all_histogram'], sort_keys=True)}` | "
            f"`{json.dumps(row['long_histogram'], sort_keys=True)}` |"
        )
    lines.extend([
        "",
        "## Direction Mix",
        "| expert | state | long_fraction | flat_fraction | n |",
        "|---|---|---:|---:|---:|",
    ])
    for expert_id, row in payload["direction_mix"].items():
        overall = row["overall"]
        lines.append(
            f"| {expert_id} | overall | {overall['long_fraction']:.4f} | "
            f"{overall['flat_fraction']:.4f} | {overall['n']} |"
        )
        for state, state_row in row["by_state"].items():
            lines.append(
                f"| {expert_id} | {state} | {state_row['long_fraction']:.4f} | "
                f"{state_row['flat_fraction']:.4f} | {state_row['n']} |"
            )
    lines.extend([
        "",
        "## Score Margin",
        "| threshold | within_0.10_count | within_0.10_fraction |",
        "|---:|---:|---:|",
    ])
    for threshold, row in payload["score_margin"]["thresholds"].items():
        lines.append(f"| {threshold} | {row['within_0.10_count']} | {row['within_0.10_fraction']:.4f} |")
    lines.extend([
        "",
        f"- score_quantiles: `{json.dumps(payload['score_margin']['quantiles'], sort_keys=True)}`",
        "",
        "## Expert-State Evidence",
        "| state | expert | non_default | safe_default | data_unavailable | parse_error | timeout | passes_min_support |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for state, experts in payload["expert_state_evidence"].items():
        for expert_id, row in experts.items():
            reasons = row["safe_default_reasons"]
            lines.append(
                f"| {state} | {expert_id} | {row['non_default_evidence_count']} | "
                f"{row['safe_default_count']} | {reasons.get('data_unavailable', 0)} | "
                f"{reasons.get('parse_error', 0)} | {reasons.get('timeout', 0)} | "
                f"{row['passes_min_support']} |"
            )
    lines.extend([
        "",
        f"- fallback_only_states: {json.dumps(payload['fallback_only_states'])}",
        "",
        "## Window Preflight",
        f"- window_pass: {payload.get('window_preflight', {}).get('window_pass')}",
        f"- recommendation: {payload.get('window_preflight', {}).get('recommendation')}",
        f"- state_support: `{json.dumps(payload.get('window_preflight', {}).get('state_support', {}), sort_keys=True)}`",
        "",
        "## Parallel Serial Check",
        f"- enabled: {payload['parallel_serial_check'].get('enabled')}",
        f"- passed: {payload['parallel_serial_check'].get('passed')}",
    ])
    lines.extend([
        "",
        "## Market State Coverage",
        f"- states_seen: {payload['market_state_coverage']['states_seen']}",
        f"- days_per_state: {json.dumps(payload['market_state_coverage']['days_per_state'], sort_keys=True)}",
        (
            "- episode_counts_by_state_action: "
            f"{json.dumps(payload['market_state_coverage']['episode_counts_by_state_action'], sort_keys=True)}"
        ),
        "",
        "## Technical Unavailable",
        f"- insufficient_trailing_history: {payload['technical_unavailable']['insufficient_trailing_history']}",
        f"- signal_adapter_error: {payload['technical_unavailable']['signal_adapter_error']}",
        f"- unknown: {payload['technical_unavailable']['unknown']}",
        "",
        "## News Health",
        f"- raw_item_count: {payload['news_health']['raw_item_count']}",
        f"- dedup_item_count: {payload['news_health']['dedup_item_count']}",
        f"- ticker_day_coverage_fraction: {payload['news_health']['ticker_day_coverage_fraction']:.4f}",
        f"- nlp_model_name: {payload['news_health']['nlp_model_name']}",
        f"- nlp_model_revision: {payload['news_health']['nlp_model_revision']}",
        f"- nlp_config_hash: {payload['news_health']['nlp_config_hash']}",
        f"- nlp_item_count: {payload['news_health']['nlp_item_count']}",
        f"- nlp_cache_hits: {payload['news_health']['nlp_cache_hits']}",
        f"- nlp_cache_misses: {payload['news_health']['nlp_cache_misses']}",
        "",
        "## Pass Gates",
    ])
    for key, value in payload["pass_gates"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def write_reports(payload: Mapping[str, object], output_base: str | Path) -> tuple[Path, Path]:
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(payload_to_markdown(payload), encoding="utf-8")
    return json_path, md_path


def _config(args: argparse.Namespace, tickers: Sequence[str]) -> BacktestConfig:
    return BacktestConfig(
        date_from=args.start,
        date_to=args.end,
        initial_cash=args.initial_cash,
        tickers=tuple(tickers),
        execution_cost_bps_per_side=args.headline_cost_bps_per_side,
        preload_date_from=args.preload_start,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start >= args.end:
        raise ValueError("--start must be before --end")
    _normalise_expert_model_identity(args)
    if args.parallel_experts is None:
        args.parallel_experts = str(args.expert_backend).strip() == "codex_sdk"
    _require_no_reserved_overlap(args.start, args.end, args.reserved_test_period)
    args.preload_start = _preload_start(args.start, args.preload_calendar_days)
    data_root = str(resolve_data_root(args.data_root))
    factor_artifact_tickers = _factor_artifact_tickers(args.alpha158_factors_db)
    selection = select_fixed_universe(
        data_root,
        args.start,
        args.end,
        n=args.names,
        allowed_tickers=factor_artifact_tickers,
    )
    loader = load_panel(data_root, args.preload_start, args.end, tickers=selection.tickers, modalities=("price",))
    if args.preflight_only:
        market_states = classify_market_states_only(
            loader,
            _config(args, selection.tickers),
            market_state_config=MarketStateConfig(
                vol_window=args.vol_window,
                percentile_window=args.percentile_window,
            ),
        )
        window_preflight = build_window_preflight(
            market_states=market_states,
            tickers=selection.tickers,
            min_support=int(args.reflection_min_support),
            start=args.start,
            end=args.end,
            reserved_periods=args.reserved_test_period,
        )
        payload = {
            "mode": "DIAGNOSTIC_WINDOW_PREFLIGHT",
            "manifest": {
                "mode": "DIAGNOSTIC_WINDOW_PREFLIGHT",
                "signal_only": True,
                "performance_metrics_computed": False,
                "window_role": str(args.window_role),
                "data_root": data_root,
                "universe_label": selection.label,
                "tickers": list(selection.tickers),
                "diagnostic_window": {"start": args.start.isoformat(), "end": args.end.isoformat()},
                "reserved_test_periods": list(args.reserved_test_period),
                "reflection_min_support": int(args.reflection_min_support),
                "factor_artifact_tickers_constraint": list(factor_artifact_tickers or []),
                "news_sentiment_enabled": False,
            },
            "window_preflight": window_preflight,
        }
        json_path, md_path = write_preflight_report(payload, args.output)
        print(f"WROTE {json_path} and {md_path}", flush=True)
        return 0
    news_sentiment_backend = None
    signal_sources = build_expert_signal_sources(
        data_root=data_root,
        start=args.preload_start,
        end=args.end,
        tickers=selection.tickers,
        alpha158_factors_db=args.alpha158_factors_db,
        qlib_provider_uri=args.qlib_provider_uri,
        news_sentiment_backend=news_sentiment_backend,
        news_sentiment_cache_path=args.news_sentiment_cache,
        include_news=not args.no_news,
        strict_news=args.strict_news,
    )
    built_experts = build_experts(
        args.expert_backend,
        model=args.expert_model,
        model_revision=args.expert_model_revision,
    )
    loop_result = run_signal_only_diagnostic(
        loader,
        _config(args, selection.tickers),
        built_experts,
        market_state_config=MarketStateConfig(
            vol_window=args.vol_window,
            percentile_window=args.percentile_window,
        ),
        signal_sources=signal_sources,
        decision_threshold=float(args.decision_thresholds[0]),
        progress_callback=_progress_printer(enabled=bool(args.progress)),
        parallel_expert_calls=bool(args.parallel_experts),
        parallel_serial_check_days=int(args.parallel_serial_check_days),
    )
    window_preflight = build_window_preflight(
        market_states=loop_result.market_states,
        tickers=selection.tickers,
        min_support=int(args.reflection_min_support),
        start=args.start,
        end=args.end,
        reserved_periods=args.reserved_test_period,
    )
    payload = build_diagnostic_payload(
        loop_result,
        manifest={
            "mode": "DIAGNOSTIC",
            "signal_only": True,
            "performance_metrics_computed": False,
            "window_role": str(args.window_role),
            "data_root": data_root,
            "universe_label": selection.label,
            "tickers": list(selection.tickers),
            "diagnostic_window": {"start": args.start.isoformat(), "end": args.end.isoformat()},
            "reserved_test_periods": list(args.reserved_test_period),
            "window_preflight": window_preflight,
            "decision_thresholds": [float(x) for x in args.decision_thresholds],
            "preload": {
                "calendar_days": int(args.preload_calendar_days),
                "date_from": args.preload_start.isoformat(),
            },
            "cost_protocol": {
                "model": "bps_per_side_on_traded_notional",
                "headline_bps_per_side": float(args.headline_cost_bps_per_side),
            },
            "reflection_min_support": int(args.reflection_min_support),
            "min_support": int(args.reflection_min_support),
            "expert_backend": args.expert_backend,
            "expert_model": args.expert_model,
            "expert_model_revision": args.expert_model_revision,
            "parallel_expert_calls": bool(args.parallel_experts),
            **expert_backend_metadata(
                args.expert_backend,
                model=args.expert_model,
                model_revision=args.expert_model_revision,
            ),
            "alpha158_factors_db": args.alpha158_factors_db,
            "factor_artifact_tickers_constraint": list(factor_artifact_tickers or []),
            "qlib_provider_uri": args.qlib_provider_uri,
            "news_enabled": not args.no_news,
            "strict_news": bool(args.strict_news),
            "news_sentiment_enabled": False,
            "news_sentiment_cache": args.news_sentiment_cache,
            "news_sentiment_model": None,
        },
    )
    json_path, md_path = write_reports(payload, args.output)
    print(f"WROTE {json_path} and {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
