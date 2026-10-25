"""Train/freeze/holdout experiment for expert-reflection memory."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, replace
from datetime import date as Date
from statistics import fmean, stdev
from typing import Mapping, Optional, Sequence

from trading_agent.contracts import BacktestConfig
from trading_agent.market import _to_date
from trading_agent.memory.expert_weights import GLOBAL_MARKET_STATE, EXPERT_IDS
from trading_agent.memory.expert_scorecard import (
    DEFAULT_TEMPERATURE,
    SOFTMAX_TEMPERATURE,
    ExpertScorecardStore,
    snapshot_payload,
)
from trading_agent.memory.expert_error import (
    expert_error_summary_to_dict,
    summarize_expert_errors,
)
from trading_agent.memory.metrics import deflated_sharpe_ratio
from trading_agent.expert_context import ExpertSignalSources
from trading_agent.memory.expert_reflection import DeterministicExpertReflection
from trading_agent.runner import ExpertLoopResult, run_expert_loop


@dataclass(frozen=True)
class ExpertWeightArmResult:
    name: str
    loop_result: ExpertLoopResult
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class ExpertWeightHoldoutResult:
    train_result: ExpertLoopResult
    global_train_result: ExpertLoopResult
    validation_arms: Mapping[str, ExpertWeightArmResult]
    arms: Mapping[str, ExpertWeightArmResult]
    split_protocol: Mapping[str, object]
    validation_arm_comparison_vs_equal_weight: Mapping[str, Mapping[str, float]]
    arm_comparison_vs_equal_weight: Mapping[str, Mapping[str, float]]
    research_claims: Mapping[str, object]
    system_completion: Mapping[str, object]
    market_state_coverage: Mapping[str, object]
    per_state_arm_metrics: Mapping[str, Mapping[str, Mapping[str, float]]]
    trial_count: int
    selected_decision_threshold: float
    selected_temperature: float
    validation_threshold_sweep: Mapping[str, Mapping[str, object]]
    freeze_ok: bool
    reflection_trial_audit: Mapping[str, object]
    expert_error_by_state: Mapping[str, object]
    weight_adjustments_by_state: Mapping[str, object]
    expert_adjustment_audit: Sequence[Mapping[str, object]]
    expert_ablation_by_state: Mapping[str, Mapping[str, Mapping[str, float]]]
    memory_snapshot: Mapping[str, object]
    arm_memory_snapshots: Mapping[str, object]
    configuration_trial_count: int
    candidate_audit_count: int
    # Terminal-verification report (None when verify_gate=False): per-store coverage +
    # per-cell stats + deltas cleared (treatment receipt, research_review_2026-07-16).
    verify_gate_report: Mapping[str, object] | None = None
    # E-SKILL: train-window text lessons distilled at the freeze point (audit +
    # the fed arm's input). Never read by aggregation or the numeric memory.
    lessons_by_state: Mapping[str, Sequence[str]] | None = None


@dataclass(frozen=True)
class _CollapsedReflectionEpisode:
    decision_date: Date
    market_state: str
    attribution: Mapping[str, float]
    signal_attribution: Mapping[str, float]
    expert_errors: tuple = ()
    action: str = "long"
    ticker: str | None = None


def _global_episode_view(episode) -> _CollapsedReflectionEpisode:
    return _CollapsedReflectionEpisode(
        decision_date=episode.decision_date,
        market_state=GLOBAL_MARKET_STATE,
        attribution=episode.attribution,
        signal_attribution=episode.signal_attribution,
        expert_errors=tuple(
            replace(record, market_state=GLOBAL_MARKET_STATE)
            for record in getattr(episode, "expert_errors", ())
        ),
        action=getattr(episode, "action", "long"),
        ticker=getattr(episode, "ticker", None),
    )


class _GlobalReflectionAdapter:
    """Run the same reflection backend with every episode collapsed into one bucket."""

    def __init__(self, reflection, *, min_support: int):
        self._reflection = reflection or DeterministicExpertReflection(min_support=min_support)

    def reflect(self, episodes, created_at: Date, participation_rate: float):
        collapsed = [_global_episode_view(episode) for episode in episodes]
        proposals = self._reflection.reflect(collapsed, created_at, participation_rate)
        return [
            replace(entry, market_state=GLOBAL_MARKET_STATE)
            if entry.market_state != GLOBAL_MARKET_STATE
            else entry
            for entry in proposals
        ]


class _StateScopedReflectionAdapter:
    """Drop 'global'-state proposals from the per-state training run.

    The market_state_memory arm must learn only per-state entries; an injected
    backend (e.g. the LLM reflection ablation) could otherwise emit a 'global'
    entry that get_weights would then apply to every state, blurring the arm
    distinction.
    """

    def __init__(self, reflection, *, min_support: int):
        self._reflection = reflection or DeterministicExpertReflection(min_support=min_support)

    def reflect(self, episodes, created_at: Date, participation_rate: float):
        proposals = self._reflection.reflect(episodes, created_at, participation_rate)
        return [
            entry for entry in proposals
            if entry.market_state != GLOBAL_MARKET_STATE
        ]


def _copy_config(config: BacktestConfig, date_from: Date, date_to: Date) -> BacktestConfig:
    from dataclasses import replace

    return replace(config, date_from=date_from, date_to=date_to)


def _cost_config(config: BacktestConfig) -> dict[str, object]:
    return {
        # A positive bps cost REPLACES the per-share commission model in the
        # execution adapter; record which model is actually charged.
        "cost_model": (
            "bps_per_side_replaces_commission"
            if config.execution_cost_bps_per_side > 0
            else "per_share_commission"
        ),
        "execution_cost_bps_per_side": float(config.execution_cost_bps_per_side),
        "commission_per_share": float(config.commission_per_share),
        "min_commission": float(config.min_commission),
        "max_commission_rate": float(config.max_commission_rate),
        "slippage_perc": float(config.slippage_perc),
        "slippage_impact": float(config.slippage_impact),
        "liquidity_cap_pct": float(config.liquidity_cap_pct),
    }


def _normalise_cost_value(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return float(value)


def assert_arm_cost_configs_match(configs: Mapping[str, Mapping[str, object]]) -> None:
    """Refuse comparisons where arms use different transaction-cost assumptions."""
    iterator = iter(configs.items())
    try:
        first_name, first = next(iterator)
    except StopIteration:
        return
    first_norm = {str(k): _normalise_cost_value(v) for k, v in first.items()}
    for name, config in iterator:
        norm = {str(k): _normalise_cost_value(v) for k, v in config.items()}
        if norm != first_norm:
            raise ValueError(
                f"arm cost configuration mismatch: {first_name}={first_norm}, {name}={norm}"
            )


def _effective_trial_count(configuration_trial_count: int, override: int | None) -> int:
    """The deflated-Sharpe trial budget. An override may only ENLARGE it
    (pre-registered cumulative accounting across experiment families);
    shrinking it would inflate every DSR and is refused."""
    if override is None:
        return configuration_trial_count
    if int(override) < configuration_trial_count:
        raise ValueError(
            f"dsr_trial_count={override} is below the run's own configuration "
            f"count {configuration_trial_count}; the trial budget may only grow"
        )
    return int(override)


def _first_test_date(dates: Sequence[Date], boundary: Date, embargo_days: int) -> Date:
    idx = bisect.bisect_right(dates, _to_date(boundary))
    test_idx = idx + embargo_days
    if test_idx >= len(dates):
        raise ValueError("embargo pushes the test window beyond available data")
    return dates[test_idx]


def _returns(equity_curve: Sequence[tuple]) -> list[float]:
    out: list[float] = []
    for (_, prev), (_, cur) in zip(equity_curve, equity_curve[1:]):
        if prev:
            out.append(cur / prev - 1.0)
    return out


def _max_drawdown(equity_curve: Sequence[tuple]) -> float:
    peak = None
    worst = 0.0
    for _, equity in equity_curve:
        peak = equity if peak is None else max(peak, equity)
        if peak:
            worst = min(worst, equity / peak - 1.0)
    return abs(worst)


def _metrics(result: ExpertLoopResult, trial_count: int) -> dict[str, float]:
    equity = list(result.backtest.equity_curve)
    if len(equity) >= 2 and equity[0][1]:
        total_return = equity[-1][1] / equity[0][1] - 1.0
    else:
        total_return = 0.0
    rets = _returns(equity)
    # Sample stdev (ddof=1) to match the deflated Sharpe's internal estimate and the
    # standard annualised-Sharpe convention; population stdev would disagree with the
    # DSR reported in the same metrics dict on short windows.
    if len(rets) > 1 and stdev(rets) > 0:
        sharpe = fmean(rets) / stdev(rets) * math.sqrt(252)
    else:
        sharpe = 0.0
    dsr = deflated_sharpe_ratio(
        rets,
        n_trials=max(2, int(trial_count)),
        var_sr=1.0 / max(1, len(rets) - 1),
    )
    # Certainty-equivalent return CE = mean - gamma * variance of daily returns, with the
    # same risk-aversion gamma=1 used in the scorecard retrieval (Chapter 3).
    if len(rets) > 1:
        certainty_equivalent = fmean(rets) - 1.0 * (stdev(rets) ** 2)
    else:
        certainty_equivalent = fmean(rets) if rets else 0.0
    fills = list(result.backtest.fills)
    traded_notional = sum(abs(f.quantity * f.fill_price) for f in fills)
    equity_values = [e for _, e in equity]
    mean_equity = fmean(equity_values) if equity_values else 0.0
    turnover = traded_notional / mean_equity if mean_equity else 0.0
    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "deflated_sharpe_ratio": None if dsr is None else float(dsr),
        "sharpe_trial_count": float(trial_count),
        "certainty_equivalent": float(certainty_equivalent),
        "max_drawdown": float(_max_drawdown(equity)),
        "turnover": float(turnover),
        "traded_notional": float(traded_notional),
        "n_fills": float(len(fills)),
        "participation_rate": float(result.participation_rate),
        "trial_count": float(trial_count),
        "learned_adjustment_count": float(result.n_learned),
    }


def _normalise_threshold_grid(values: Sequence[float] | None) -> tuple[float, ...]:
    raw = values if values is not None else (0.1, 0.2, 0.3, 0.4, 0.5)
    out = []
    for value in raw:
        threshold = float(value)
        # A = sum w*v in [-1, 1]; a negative entry threshold is the
        # participation-biased rule (long unless strongly bearish).
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("decision thresholds must be in [-1, 1]")
        out.append(round(threshold, 6))
    unique = tuple(sorted(set(out)))
    if not unique:
        raise ValueError("decision threshold grid must be non-empty")
    return unique


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):.4f}"


def _temperature_key(temperature: float) -> str:
    return f"{float(temperature):.5f}"


def _normalise_temperature_grid(
    values: Sequence[float] | None, *, standardize: bool = True
) -> tuple[float, ...]:
    """Softmax-temperature grid for validation. Defaults to a single value matching the
    retrieval mode (0.5 standardized, 0.001 raw), so a run that does not tune temperature
    is unchanged; the real study passes the mode's three-point grid. Because the scorecard
    moments are temperature-invariant, each grid temperature is a numeric re-read of the
    frozen store."""
    raw = values if values is not None else (
        (DEFAULT_TEMPERATURE,) if standardize else (SOFTMAX_TEMPERATURE,)
    )
    out = [round(float(v), 8) for v in raw if float(v) > 0.0]
    unique = tuple(sorted(set(out)))
    if not unique:
        raise ValueError("temperature grid must be non-empty and positive")
    return unique


def _retuned(store: ExpertScorecardStore, temperature: float) -> ExpertScorecardStore:
    """Re-read a frozen scorecard at a different softmax temperature (moments unchanged)."""
    return ExpertScorecardStore.from_snapshot(store.snapshot(), temperature=temperature)


def _select_threshold(validation_by_threshold: Mapping[float, Mapping[str, ExpertWeightArmResult]]) -> float:
    ranked = []
    for threshold, arms in validation_by_threshold.items():
        primary = arms.get("market_state_memory")
        if primary is None:
            continue
        metrics = primary.metrics
        ranked.append((
            float(metrics.get("sharpe", 0.0)),
            float(metrics.get("total_return", 0.0)),
            float(metrics.get("participation_rate", 0.0)),
            -float(threshold),
            float(threshold),
        ))
    if not ranked:
        return 0.5
    return max(ranked)[-1]


def _arm_comparison_vs_equal_weight(
    arms: Mapping[str, ExpertWeightArmResult],
) -> dict[str, dict[str, float]]:
    baseline = arms["equal_weight"].metrics
    keys = (
        "total_return",
        "sharpe",
        "max_drawdown",
        "turnover",
        "participation_rate",
        "learned_adjustment_count",
    )
    out: dict[str, dict[str, float]] = {}
    for name, arm in arms.items():
        if name == "equal_weight":
            continue
        out[name] = {
            f"{key}_delta": float(arm.metrics.get(key, 0.0) - baseline.get(key, 0.0))
            for key in keys
        }
    return out


def _metric_deltas(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> dict[str, float]:
    keys = (
        "total_return",
        "sharpe",
        "max_drawdown",
        "turnover",
        "participation_rate",
        "learned_adjustment_count",
    )
    return {f"{key}_delta": float(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys}


#: Documented ablation semantics: the removed expert is muted (emits no views) but
#: the weight projection is unchanged, so its weight mass stays
#: allocated at the floor rather than being redistributed to the remaining three.
ABLATION_SEMANTICS: str = "expert_muted_weight_mass_retained"


def _best_expert_ablation(
    expert_ablation_by_state: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, object]:
    best = {
        "expert_id": None,
        "market_state": None,
        "signal_component_drop": 0.0,
        "participation_rate_drop": 0.0,
    }
    for expert_id, states in expert_ablation_by_state.items():
        for state, metrics in states.items():
            drop = float(metrics.get("signal_component_drop", 0.0))
            if best["expert_id"] is None or drop > float(best["signal_component_drop"]):
                best = {
                    "expert_id": expert_id,
                    "market_state": state,
                    "signal_component_drop": drop,
                    "participation_rate_drop": float(metrics.get("participation_rate_drop", 0.0)),
                }
    return best


def _research_claims(
    arms: Mapping[str, ExpertWeightArmResult],
    arm_comparison_vs_equal_weight: Mapping[str, Mapping[str, float]],
    expert_ablation_by_state: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    freeze_ok: bool,
) -> dict[str, object]:
    state_arm = arms["market_state_memory"]
    global_arm = arms["global_memory"]
    return {
        "reflection_contribution": {
            "comparison": "market_state_memory_vs_equal_weight",
            **dict(arm_comparison_vs_equal_weight.get("market_state_memory", {})),
        },
        "regime_conditioning_contribution": {
            "comparison": "market_state_memory_vs_global_memory",
            **_metric_deltas(state_arm.metrics, global_arm.metrics),
        },
        "best_expert_ablation": {
            **_best_expert_ablation(expert_ablation_by_state),
            "semantics": ABLATION_SEMANTICS,
        },
        "frozen_test_protocol": {
            "freeze_ok": bool(freeze_ok),
            "allow_reflection": False,
            "memory_status": "frozen",
        },
    }


def _system_completion(
    train_result: ExpertLoopResult,
    validation_arms: Mapping[str, ExpertWeightArmResult],
    arms: Mapping[str, ExpertWeightArmResult],
    memory_snapshot: Mapping[str, object],
    expert_error_by_state: Mapping[str, object],
    expert_ablation_by_state: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    freeze_ok: bool,
) -> dict[str, object]:
    required_arms = {"equal_weight", "global_memory", "market_state_memory"}
    # The online (self-improving) arm reflects during the test window BY DESIGN, so it is
    # exempt from the frozen-test guarantee; the freeze guarantee is about the frozen arms
    # (and validation) not learning on the evaluation windows.
    frozen_test_arms = [
        arm for name, arm in arms.items() if name != "market_state_memory_online"
    ]
    test_reflection_disabled = all(
        arm.loop_result.n_reflection_rounds == 0
        for arm in list(validation_arms.values()) + frozen_test_arms
    )
    checks = {
        "train_has_active_episodes": bool(train_result.n_active_dates > 0 and train_result.episodes),
        "train_reflection_ran": bool(train_result.n_reflection_rounds > 0),
        "freeze_ok": bool(freeze_ok),
        "test_reflection_disabled": bool(test_reflection_disabled),
        "memory_snapshot_frozen": memory_snapshot.get("status") == "frozen",
        "all_required_arms_present": required_arms <= set(arms),
        "validation_window_read_only": all(
            arm.loop_result.n_reflection_rounds == 0 for arm in validation_arms.values()
        ),
        "expert_error_reported": bool(expert_error_by_state.get("train")),
        "per_state_ablation_reported": set(expert_ablation_by_state) == set(EXPERT_IDS)
        and all(bool(states) for states in expert_ablation_by_state.values()),
    }
    return {
        "research_system_complete": all(checks.values()),
        "scope": "research_system",
        "external_paper_validation_required": False,
        "ready_for_defense_sections": [
            "data_observation",
            "market_state_classifier",
            "four_experts",
            "expert_error_analysis",
            "reflection_memory",
            "train_freeze_validation_test",
            "three_arm_evaluation",
            "per_state_ablation",
        ],
        "checks": checks,
    }


def _state_counts(result: ExpertLoopResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in result.episodes:
        counts[episode.market_state] = counts.get(episode.market_state, 0) + 1
    return counts


def _market_state_coverage(
    train_result: ExpertLoopResult,
    arms: Mapping[str, ExpertWeightArmResult],
) -> dict[str, object]:
    return {
        "train": _state_counts(train_result),
        "arms": {name: _state_counts(arm.loop_result) for name, arm in arms.items()},
    }


def _per_state_metrics(result: ExpertLoopResult) -> dict[str, dict[str, float]]:
    grouped: dict[str, list] = {}
    for episode in result.episodes:
        grouped.setdefault(episode.market_state, []).append(episode)

    out: dict[str, dict[str, float]] = {}
    for state, episodes in sorted(grouped.items()):
        n = len(episodes)
        # Flat episodes are counterfactual learning rows; component means are taken
        # over long-action episodes only so flat rows never dilute them. The flat
        # share stays visible through episode_count vs long_episode_count.
        long_episodes = [ep for ep in episodes if ep.action == "long"]
        market_returns = [float(ep.attribution.get("market", 0.0)) for ep in long_episodes]
        signal_components = [float(ep.attribution.get("signal", 0.0)) for ep in long_episodes]
        out[state] = {
            "episode_count": float(n),
            "long_episode_count": float(len(long_episodes)),
            "mean_market_return": float(fmean(market_returns)) if market_returns else 0.0,
            "mean_signal_component": float(fmean(signal_components)) if signal_components else 0.0,
            "participation_rate": float(len(long_episodes) / n) if n else 0.0,
        }
    return out


def _per_state_arm_metrics(
    arms: Mapping[str, ExpertWeightArmResult],
) -> dict[str, dict[str, dict[str, float]]]:
    return {name: _per_state_metrics(arm.loop_result) for name, arm in arms.items()}


def _error_summary_from_result(result: ExpertLoopResult) -> dict:
    records = [record for ep in result.episodes for record in getattr(ep, "expert_errors", ())]
    return expert_error_summary_to_dict(summarize_expert_errors(records))


def _weight_adjustments_by_state(store: ExpertScorecardStore) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for entry in store.learned():
        out.setdefault(entry.market_state, []).append({
            "expert_id": entry.expert_id,
            "weight_delta": float(entry.weight_delta),
            "rationale": entry.rationale,
        })
    return out


def _expert_adjustment_audit(
    store: ExpertScorecardStore,
    train_error_summary: Mapping[str, object],
    per_state_arm_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> list[dict[str, object]]:
    equal_metrics = per_state_arm_metrics.get("equal_weight", {})
    state_metrics = per_state_arm_metrics.get("market_state_memory", {})
    rows: list[dict[str, object]] = []
    for entry in store.learned():
        train_state = train_error_summary.get(entry.market_state, {})
        train_expert = train_state.get(entry.expert_id, {}) if isinstance(train_state, dict) else {}
        equal_state = equal_metrics.get(entry.market_state, {})
        memory_state = state_metrics.get(entry.market_state, {})
        rows.append({
            "market_state": entry.market_state,
            "expert_id": entry.expert_id,
            "weight_delta": float(entry.weight_delta),
            "rationale": entry.rationale,
            "train_error_n": int(train_expert.get("n", 0)) if isinstance(train_expert, dict) else 0,
            "train_accuracy": float(train_expert.get("accuracy", 0.0)) if isinstance(train_expert, dict) else 0.0,
            "train_mean_error_score": (
                float(train_expert.get("mean_error_score", 0.0)) if isinstance(train_expert, dict) else 0.0
            ),
            "train_overconfidence_loss": (
                float(train_expert.get("overconfidence_loss", 0.0)) if isinstance(train_expert, dict) else 0.0
            ),
            "train_missed_opportunity_loss": (
                float(train_expert.get("missed_opportunity_loss", 0.0)) if isinstance(train_expert, dict) else 0.0
            ),
            "state_episode_count": float(memory_state.get("episode_count", 0.0)),
            "state_signal_component_delta": float(
                memory_state.get("mean_signal_component", 0.0)
                - equal_state.get("mean_signal_component", 0.0)
            ),
            "state_participation_delta": float(
                memory_state.get("participation_rate", 0.0)
                - equal_state.get("participation_rate", 0.0)
            ),
        })
    return rows


def _expert_ablation_by_state(
    full_metrics: Mapping[str, Mapping[str, float]],
    ablated: Mapping[str, ExpertWeightArmResult],
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for expert_id, arm in ablated.items():
        ablated_metrics = _per_state_metrics(arm.loop_result)
        states = sorted(set(full_metrics) | set(ablated_metrics))
        out[expert_id] = {}
        for state in states:
            full_state = full_metrics.get(state, {})
            ablated_state = ablated_metrics.get(state, {})
            full_signal = float(full_state.get("mean_signal_component", 0.0))
            ablated_signal = float(ablated_state.get("mean_signal_component", 0.0))
            full_participation = float(full_state.get("participation_rate", 0.0))
            ablated_participation = float(ablated_state.get("participation_rate", 0.0))
            out[expert_id][state] = {
                "full_episode_count": float(full_state.get("episode_count", 0.0)),
                "ablated_episode_count": float(ablated_state.get("episode_count", 0.0)),
                "full_mean_signal_component": full_signal,
                "ablated_mean_signal_component": ablated_signal,
                "signal_component_drop": float(full_signal - ablated_signal),
                "full_participation_rate": full_participation,
                "ablated_participation_rate": ablated_participation,
                "participation_rate_drop": float(full_participation - ablated_participation),
            }
    return out


def run_expert_weight_holdout(
    loader,
    config: BacktestConfig,
    experts: Sequence,
    *,
    boundary: Date,
    validation_boundary: Date | None = None,
    embargo_days: int = 5,
    market_state_config=None,
    reflection=None,
    reflect_every: int = 21,
    reflection_window: int = 63,
    reflection_min_support: int = 5,
    decision_threshold_grid: Sequence[float] | None = None,
    temperature_grid: Sequence[float] | None = None,
    signal_sources: ExpertSignalSources | None = None,
    attribution_mode: str = "raw",
    attribution_horizon: int = 1,
    bounded_memory: bool = True,
    use_reflection: bool = True,
    certified_only_memory: bool = False,
    standardize_memory: bool = False,
    shrink_to_global: bool = False,
    exit_threshold: float | None = None,
    memory_half_life: float | None = None,
    variance_penalty: float | None = None,
    memory_score: str = "ewvar",
    memory_shrinkage: str = "none",
    position_sizing: str = "binary",
    stale_view_abstain: int = 0,
    dsr_trial_count: int | None = None,
    market_proxy=None,
    parallel_expert_calls: bool = False,
    verify_gate: bool = False,
    verify_alpha: float = 0.10,
    verify_correction: str = "holm",
    verify_persistence: bool = False,
    verify_min_effect: float = 0.0,
    verify_oos: bool = False,
    verify_kfold: int = 0,
    verify_block: int | None = None,
    confidence_weighted_vote: bool = False,
    soft_abstain_weight: float = 0.0,
    decision_cadence: int = 1,
) -> ExpertWeightHoldoutResult:
    """Run train-time reflection, freeze memory, then evaluate three test arms.

    ``attribution_mode`` is threaded to every internal run so the whole system
    can be compared under market-adjusted (default) versus raw scoring (E2).
    ``bounded_memory=False`` builds the learned stores without the cumulative
    delta clip, for the bounded-vs-unbounded ablation (E3).
    ``use_reflection=False`` ignores the reflection deltas at retrieval (the
    numeric-only ablation); the scorecard moments are unchanged.
    """
    threshold_grid = _normalise_threshold_grid(decision_threshold_grid)
    temperature_grid = _normalise_temperature_grid(temperature_grid, standardize=standardize_memory)
    train_decision_threshold = 0.5 if 0.5 in threshold_grid else max(threshold_grid)
    dates = [_to_date(d) for d in loader.get_date_range()]
    if not dates:
        raise ValueError("no dates available")
    boundary = _to_date(boundary)
    train_boundary = _to_date(validation_boundary) if validation_boundary is not None else boundary
    if train_boundary > boundary:
        raise ValueError("validation_boundary must be on or before boundary")
    test_start = _first_test_date(dates, boundary, embargo_days)
    if test_start > _to_date(config.date_to):
        raise ValueError("test window starts after config.date_to")

    train_cfg = _copy_config(config, _to_date(config.date_from), train_boundary)
    test_cfg = _copy_config(config, test_start, _to_date(config.date_to))
    validation_cfg = None
    validation_start = None
    if validation_boundary is not None:
        validation_start = _first_test_date(dates, train_boundary, embargo_days)
        if validation_start > boundary:
            raise ValueError("validation window starts after boundary")
        validation_cfg = _copy_config(config, validation_start, boundary)

    # The scorecard tilt threshold is a fixed 20 per-name episodes (store default), distinct
    # from the reflection evidence threshold (reflection_min_support) and the evaluation-layer
    # underpowered threshold. Per-name episodes accumulate ~n_tickers faster than active dates.
    _prior_kwargs = {}
    if memory_half_life is not None:
        _prior_kwargs["half_life"] = float(memory_half_life)
    if variance_penalty is not None:
        _prior_kwargs["variance_penalty"] = float(variance_penalty)
    # Family E1: hit-rate log-odds retrieval; ctor validates the mode and the
    # snapshot round-trip carries it through every validation/test re-read.
    _prior_kwargs["score_mode"] = str(memory_score)
    _prior_kwargs["shrinkage"] = str(memory_shrinkage)
    _prior_kwargs["overlap"] = int(attribution_horizon)
    state_store = ExpertScorecardStore(
        scope="state", bound_reflection=bounded_memory, use_reflection=use_reflection,
        certified_only=certified_only_memory,
        standardize=standardize_memory, shrink_to_global=shrink_to_global,
        **_prior_kwargs,
    )
    train_result = run_expert_loop(
        loader,
        train_cfg,
        experts,
        market_state_config=market_state_config,
        weight_store=state_store,
        reflection=_StateScopedReflectionAdapter(reflection, min_support=reflection_min_support),
        reflect_every=reflect_every,
        reflection_window=reflection_window,
        reflection_min_support=reflection_min_support,
        decision_threshold=train_decision_threshold,
        exit_threshold=exit_threshold, position_sizing=position_sizing,
        stale_view_abstain=stale_view_abstain,
        allow_reflection=True,
        signal_sources=signal_sources,
        attribution_mode=attribution_mode,
        attribution_horizon=attribution_horizon,
        market_proxy=market_proxy,
        parallel_expert_calls=parallel_expert_calls,
        confidence_weighted_vote=confidence_weighted_vote,
        soft_abstain_weight=soft_abstain_weight,
        decision_cadence=decision_cadence,
        run_id="expert_weight_train",
    )

    global_store = ExpertScorecardStore(
        scope="global", bound_reflection=bounded_memory, use_reflection=use_reflection,
        certified_only=certified_only_memory,
        standardize=standardize_memory,
        **_prior_kwargs,
    )
    global_train_result = run_expert_loop(
        loader,
        train_cfg,
        experts,
        market_state_config=market_state_config,
        weight_store=global_store,
        reflection=_GlobalReflectionAdapter(reflection, min_support=reflection_min_support),
        reflect_every=reflect_every,
        reflection_window=reflection_window,
        reflection_min_support=reflection_min_support,
        decision_threshold=train_decision_threshold,
        exit_threshold=exit_threshold, position_sizing=position_sizing,
        stale_view_abstain=stale_view_abstain,
        allow_reflection=True,
        signal_sources=signal_sources,
        attribution_mode=attribution_mode,
        attribution_horizon=attribution_horizon,
        market_proxy=market_proxy,
        parallel_expert_calls=parallel_expert_calls,
        confidence_weighted_vote=confidence_weighted_vote,
        soft_abstain_weight=soft_abstain_weight,
        decision_cadence=decision_cadence,
        run_id="expert_weight_train_global",
    )
    configuration_trial_count = len(temperature_grid) * len(threshold_grid)
    candidate_audit_count = train_result.n_proposals + global_train_result.n_proposals
    trial_count = _effective_trial_count(configuration_trial_count, dsr_trial_count)

    equal_store = ExpertScorecardStore(scope="state")

    # TERMINAL verification (research_review_2026-07-16): ONE pass at freeze time, over ALL
    # training episodes, clearing every reflection delta whose cell is not statistically
    # verified (marginal proxy m, synchronized-date studentized block bootstrap, Holm).
    # Backend-agnostic: cleans whatever the maker (LLM or deterministic) wrote. Runs BEFORE
    # the online-arm snapshot and the freezes so every arm inherits the verified memory.
    verify_gate_report = None
    if verify_gate:
        from trading_agent.memory.expert_reflection import (
            apply_terminal_gate,
            delta_sign_receipt,
            marginal_series,
            terminal_verify,
        )

        # Out-of-sample confirmation (opt-in): harvest the held-out validation window's
        # attributed contributions (attribution is weight-independent, so any store works)
        # so a verified cell can be required to keep its sign on data it was not fit on.
        oos_state_series = oos_global_series = None
        if verify_oos and validation_cfg is not None:
            oos_loop = run_expert_loop(
                loader, validation_cfg, experts,
                market_state_config=market_state_config,
                weight_store=ExpertScorecardStore(scope="state"),
                reflection=reflection, reflect_every=reflect_every,
                reflection_window=reflection_window, reflection_min_support=reflection_min_support,
                decision_threshold=0.0, exit_threshold=exit_threshold, position_sizing=position_sizing,
        stale_view_abstain=stale_view_abstain, allow_reflection=False,
                signal_sources=signal_sources, attribution_mode=attribution_mode,
                attribution_horizon=attribution_horizon, market_proxy=market_proxy,
                parallel_expert_calls=parallel_expert_calls, run_id="expert_weight_verify_oos",
            )
            oos_state_series = marginal_series(oos_loop.episodes)
            oos_global_series = marginal_series(oos_loop.episodes, pool_states=True)

        # Horizon-aware block length (audit fix 2026-07-19): h-day counterfactual
        # returns sampled daily are ~h-dependent, so the default n^(1/3) block
        # (~14 here) under-covers h=21 and yields anti-conservative p-values.
        # A-priori rule: blocks must span at least twice the dependence length
        # (Politis-White guidance); h=1 keeps the default -> all prior results
        # byte-identical.
        # Block length for the moving-block bootstrap. h-day overlapping returns lose their
        # autocorrelation at lag h, so a block of h preserves the dependence; the default
        # doubles it as extra conservatism. --verify-block overrides it (prereg Amendment 9).
        verify_block = (
            int(verify_block) if verify_block
            else (max(2 * attribution_horizon, 14) if attribution_horizon > 1 else None)
        )
        state_verified, state_report = terminal_verify(
            train_result.episodes, block_len=verify_block, alpha=verify_alpha, correction=verify_correction,
            require_persistence=verify_persistence, min_effect=verify_min_effect,
            oos_series=oos_state_series, kfold=verify_kfold,
        )
        state_delta_receipt = delta_sign_receipt(state_store, state_report)
        state_cleared = apply_terminal_gate(state_store, state_verified)
        state_store.set_certified_cells(state_verified)
        global_verified, global_report = terminal_verify(
            global_train_result.episodes, pool_states=True, block_len=verify_block, alpha=verify_alpha,
            correction=verify_correction,
            require_persistence=verify_persistence, min_effect=verify_min_effect,
            oos_series=oos_global_series, kfold=verify_kfold,
        )
        global_delta_receipt = delta_sign_receipt(global_store, global_report)
        global_cleared = apply_terminal_gate(global_store, global_verified)
        global_store.set_certified_cells(global_verified)
        verify_gate_report = {
            "market_state_memory": {**state_report, "deltas_cleared": state_cleared,
                                    "delta_receipt": state_delta_receipt},
            "global_memory": {**global_report, "deltas_cleared": global_cleared,
                              "delta_receipt": global_delta_receipt},
        }

    # E-SKILL (architecture Phase 2.5): distill the training window's attributed
    # findings into compact TEXT lessons at the same freeze point as the numeric
    # memory. Train-window only, frozen alongside the scorecard; the fed arm
    # feeds these back to the experts as `prior_lessons` (input-side treatment)
    # while the numeric harness stays the instrument.
    from trading_agent.memory.expert_reflection import _compact_lessons_by_state

    lessons_by_state = _compact_lessons_by_state(train_result.episodes)

    # Writable snapshot of the trained state memory, captured BEFORE freezing, so the
    # online (self-improving) arm can keep updating through the test window. A pre-freeze
    # snapshot has frozen=False, so from_snapshot rebuilds a writable store.
    state_trained_snapshot = state_store.snapshot()

    state_store.freeze(train_boundary)
    global_store.freeze(train_boundary)
    equal_store.freeze(train_boundary)

    before = {
        "market_state_memory": state_store.snapshot(),
        "global_memory": global_store.snapshot(),
        "equal_weight": equal_store.snapshot(),
    }

    arm_stores = {
        "equal_weight": equal_store,
        "global_memory": global_store,
        "market_state_memory": state_store,
    }
    validation_arms: dict[str, ExpertWeightArmResult] = {}
    validation_by_threshold: dict[float, dict[str, ExpertWeightArmResult]] = {}
    if validation_cfg is not None:
        market = "market_state_memory"
        base_snapshot = arm_stores[market].snapshot()
        # Grid is temperature x threshold. Because the scorecard moments are
        # temperature-invariant, each temperature is a numeric re-read of the frozen store
        # (no retraining, no language-model call). The threshold sweep is reported at the
        # selected temperature; the winning temperature is selected across the outer loop.
        best: tuple | None = None
        for temperature in temperature_grid:
            by_threshold: dict[float, dict[str, ExpertWeightArmResult]] = {}
            for threshold in threshold_grid:
                store = ExpertScorecardStore.from_snapshot(base_snapshot, temperature=temperature)
                loop_result = run_expert_loop(
                    loader,
                    validation_cfg,
                    experts,
                    market_state_config=market_state_config,
                    weight_store=store,
                    reflection=reflection,
                    reflect_every=reflect_every,
                    reflection_window=reflection_window,
                    reflection_min_support=reflection_min_support,
                    decision_threshold=threshold,
        exit_threshold=exit_threshold, position_sizing=position_sizing,
        stale_view_abstain=stale_view_abstain,
                    allow_reflection=False,
                    signal_sources=signal_sources,
                    attribution_mode=attribution_mode,
                    attribution_horizon=attribution_horizon,
                    market_proxy=market_proxy,
                    parallel_expert_calls=parallel_expert_calls,
                    confidence_weighted_vote=confidence_weighted_vote,
                    soft_abstain_weight=soft_abstain_weight,
                    decision_cadence=decision_cadence,
                    run_id=(f"expert_weight_validation_{market}_"
                            f"t{_temperature_key(temperature)}_thr_{_threshold_key(threshold)}"),
                )
                by_threshold[threshold] = {market: ExpertWeightArmResult(
                    name=market, loop_result=loop_result, metrics=_metrics(loop_result, trial_count),
                )}
            sel_thr = _select_threshold(by_threshold)
            m = by_threshold[sel_thr][market].metrics
            score = (
                float(m.get("sharpe", 0.0)), float(m.get("total_return", 0.0)),
                float(m.get("participation_rate", 0.0)), -float(sel_thr), -float(temperature),
            )
            if best is None or score > best[0]:
                best = (score, temperature, sel_thr, by_threshold)
        _, selected_temperature, selected_threshold, validation_by_threshold = best
        for name, store in arm_stores.items():
            if name == market:
                validation_arms[name] = validation_by_threshold[selected_threshold][name]
                continue
            loop_result = run_expert_loop(
                loader,
                validation_cfg,
                experts,
                market_state_config=market_state_config,
                weight_store=_retuned(store, selected_temperature),
                reflection=reflection,
                reflect_every=reflect_every,
                reflection_window=reflection_window,
                reflection_min_support=reflection_min_support,
                decision_threshold=selected_threshold,
        exit_threshold=exit_threshold, position_sizing=position_sizing,
        stale_view_abstain=stale_view_abstain,
                allow_reflection=False,
                signal_sources=signal_sources,
                attribution_mode=attribution_mode,
                attribution_horizon=attribution_horizon,
                market_proxy=market_proxy,
                parallel_expert_calls=parallel_expert_calls,
                confidence_weighted_vote=confidence_weighted_vote,
                soft_abstain_weight=soft_abstain_weight,
                decision_cadence=decision_cadence,
                run_id=f"expert_weight_validation_{name}_thr_{_threshold_key(selected_threshold)}",
            )
            validation_arms[name] = ExpertWeightArmResult(
                name=name,
                loop_result=loop_result,
                metrics=_metrics(loop_result, trial_count),
            )
    else:
        selected_temperature = temperature_grid[0]
        selected_threshold = train_decision_threshold

    validation_threshold_sweep = {
        _threshold_key(threshold): {
            "selected": threshold == selected_threshold,
            "arms": {
                name: {
                    "metrics": dict(arm.metrics),
                    "n_active_dates": arm.loop_result.n_active_dates,
                    "participation_rate": arm.loop_result.participation_rate,
                }
                for name, arm in arms_for_threshold.items()
            },
        }
        for threshold, arms_for_threshold in sorted(validation_by_threshold.items())
    }

    arms: dict[str, ExpertWeightArmResult] = {}
    arm_cost_configs: dict[str, dict[str, float]] = {}
    for name, store in arm_stores.items():
        loop_result = run_expert_loop(
            loader,
            test_cfg,
            experts,
            market_state_config=market_state_config,
            weight_store=_retuned(store, selected_temperature),
            reflection=reflection,
            reflect_every=reflect_every,
            reflection_window=reflection_window,
            reflection_min_support=reflection_min_support,
            decision_threshold=selected_threshold,
        exit_threshold=exit_threshold, position_sizing=position_sizing,
        stale_view_abstain=stale_view_abstain,
            allow_reflection=False,
            signal_sources=signal_sources,
            attribution_mode=attribution_mode,
            attribution_horizon=attribution_horizon,
            market_proxy=market_proxy,
            parallel_expert_calls=parallel_expert_calls,
            confidence_weighted_vote=confidence_weighted_vote,
            soft_abstain_weight=soft_abstain_weight,
            decision_cadence=decision_cadence,
            run_id=f"expert_weight_{name}",
        )
        arms[name] = ExpertWeightArmResult(
            name=name,
            loop_result=loop_result,
            metrics=_metrics(loop_result, trial_count),
        )
        arm_cost_configs[name] = _cost_config(test_cfg)

    # Online (self-improving) arm: seed a WRITABLE store from the trained state memory
    # and keep updating it through the test window (allow_reflection=True). This is
    # causal, not look-ahead: run_expert_loop retrieves weights (memory through t-1),
    # decides, then observes the realised t->t+1 return, so day t never uses its own
    # outcome. It answers "does continual self-improvement help", complementing the
    # frozen arm's "does trained memory transfer". Same cost config as the frozen arms.
    online_store = ExpertScorecardStore.from_snapshot(
        state_trained_snapshot, temperature=selected_temperature
    )
    online_result = run_expert_loop(
        loader,
        test_cfg,
        experts,
        market_state_config=market_state_config,
        weight_store=online_store,
        reflection=reflection,
        reflect_every=reflect_every,
        reflection_window=reflection_window,
        reflection_min_support=reflection_min_support,
        decision_threshold=selected_threshold,
        exit_threshold=exit_threshold, position_sizing=position_sizing,
        stale_view_abstain=stale_view_abstain,
        allow_reflection=True,
        signal_sources=signal_sources,
        attribution_mode=attribution_mode,
        attribution_horizon=attribution_horizon,
        market_proxy=market_proxy,
        parallel_expert_calls=parallel_expert_calls,
        confidence_weighted_vote=confidence_weighted_vote,
        soft_abstain_weight=soft_abstain_weight,
        decision_cadence=decision_cadence,
        run_id="expert_weight_market_state_memory_online",
    )
    arms["market_state_memory_online"] = ExpertWeightArmResult(
        name="market_state_memory_online",
        loop_result=online_result,
        metrics=_metrics(online_result, trial_count),
    )
    arm_cost_configs["market_state_memory_online"] = _cost_config(test_cfg)
    assert_arm_cost_configs_match(arm_cost_configs)

    ablated_arms: dict[str, ExpertWeightArmResult] = {}
    for expert_id in EXPERT_IDS:
        ablated_experts = [expert for expert in experts if expert.expert_id != expert_id]
        loop_result = run_expert_loop(
            loader,
            test_cfg,
            ablated_experts,
            market_state_config=market_state_config,
            weight_store=_retuned(state_store, selected_temperature),
            reflection=reflection,
            reflect_every=reflect_every,
            reflection_window=reflection_window,
            reflection_min_support=reflection_min_support,
            decision_threshold=selected_threshold,
        exit_threshold=exit_threshold, position_sizing=position_sizing,
        stale_view_abstain=stale_view_abstain,
            allow_reflection=False,
            signal_sources=signal_sources,
            attribution_mode=attribution_mode,
            attribution_horizon=attribution_horizon,
            market_proxy=market_proxy,
            parallel_expert_calls=parallel_expert_calls,
            confidence_weighted_vote=confidence_weighted_vote,
            soft_abstain_weight=soft_abstain_weight,
            decision_cadence=decision_cadence,
            run_id=f"expert_weight_without_{expert_id}",
        )
        ablated_arms[expert_id] = ExpertWeightArmResult(
            name=f"without_{expert_id}",
            loop_result=loop_result,
            metrics=_metrics(loop_result, trial_count),
        )

    after = {
        "market_state_memory": state_store.snapshot(),
        "global_memory": global_store.snapshot(),
        "equal_weight": equal_store.snapshot(),
    }
    train_error_summary = _error_summary_from_result(train_result)
    arms_error_summary = {name: _error_summary_from_result(arm.loop_result) for name, arm in arms.items()}
    per_state_arm_metrics = _per_state_arm_metrics(arms)
    expert_ablation_by_state = _expert_ablation_by_state(
        per_state_arm_metrics.get("market_state_memory", {}),
        ablated_arms,
    )
    arm_comparison_vs_equal_weight = _arm_comparison_vs_equal_weight(arms)
    freeze_ok = before == after
    memory_snapshot = snapshot_payload(state_store, status="frozen")
    arm_memory_snapshots = {
        "market_state_memory": memory_snapshot,
        "global_memory": snapshot_payload(global_store, status="frozen"),
        "equal_weight": snapshot_payload(equal_store, status="frozen"),
        # The online arm's memory is the trained memory after continual test-window
        # updating; status reflects that it is NOT frozen (audit only).
        "market_state_memory_online": snapshot_payload(online_store, status="online_updated"),
    }
    return ExpertWeightHoldoutResult(
        train_result=train_result,
        global_train_result=global_train_result,
        validation_arms=validation_arms,
        arms=arms,
        split_protocol={
            "train_start": train_cfg.date_from.isoformat(),
            "train_end": train_cfg.date_to.isoformat(),
            "validation_start": validation_start.isoformat() if validation_start is not None else None,
            "validation_end": boundary.isoformat() if validation_cfg is not None else None,
            "test_start": test_cfg.date_from.isoformat(),
            "test_end": test_cfg.date_to.isoformat(),
            "embargo_days": int(embargo_days),
            "memory_frozen_at": train_boundary.isoformat(),
            "train_decision_threshold": float(train_decision_threshold),
            "selected_decision_threshold": float(selected_threshold),
            "selected_temperature": float(selected_temperature),
            "decision_threshold_grid": [float(x) for x in threshold_grid],
            "temperature_grid": [float(x) for x in temperature_grid],
            # Documented protocol ruling: the grid is swept on the primary
            # (market_state_memory) arm's validation Sharpe and the selected
            # threshold is then imposed on all arms; baselines do not run their
            # own search, and the DSR trial count charged to every arm is the
            # size of this grid.
            "threshold_selection_arm": "market_state_memory",
            "threshold_selection_metric": "sharpe",
        },
        validation_arm_comparison_vs_equal_weight=(
            _arm_comparison_vs_equal_weight(validation_arms) if validation_arms else {}
        ),
        arm_comparison_vs_equal_weight=arm_comparison_vs_equal_weight,
        research_claims=_research_claims(
            arms,
            arm_comparison_vs_equal_weight,
            expert_ablation_by_state,
            freeze_ok=freeze_ok,
        ),
        system_completion=_system_completion(
            train_result,
            validation_arms,
            arms,
            memory_snapshot,
            {"train": train_error_summary, "arms": arms_error_summary},
            expert_ablation_by_state,
            freeze_ok=freeze_ok,
        ),
        market_state_coverage=_market_state_coverage(train_result, arms),
        per_state_arm_metrics=per_state_arm_metrics,
        trial_count=trial_count,
        selected_decision_threshold=selected_threshold,
        selected_temperature=selected_temperature,
        validation_threshold_sweep=validation_threshold_sweep,
        freeze_ok=freeze_ok,
        reflection_trial_audit={
            "round_count": len(train_result.reflection_trial_records),
            "proposal_count": train_result.n_proposals,
            "global_proposal_count": global_train_result.n_proposals,
            "applied_count": train_result.n_applied,
            "global_applied_count": global_train_result.n_applied,
            "rounds": train_result.reflection_trial_records,
            "global_rounds": global_train_result.reflection_trial_records,
            "candidate_audit_count": candidate_audit_count,
        },
        expert_error_by_state={
            "train": train_error_summary,
            "arms": arms_error_summary,
        },
        weight_adjustments_by_state=_weight_adjustments_by_state(state_store),
        expert_adjustment_audit=_expert_adjustment_audit(
            state_store,
            train_error_summary,
            per_state_arm_metrics,
        ),
        expert_ablation_by_state=expert_ablation_by_state,
        memory_snapshot=memory_snapshot,
        arm_memory_snapshots=arm_memory_snapshots,
        configuration_trial_count=configuration_trial_count,
        candidate_audit_count=candidate_audit_count,
        verify_gate_report=verify_gate_report,
        lessons_by_state=lessons_by_state,
    )
