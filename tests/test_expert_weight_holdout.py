"""Tests for the train/freeze/holdout expert-weight experiment."""

from __future__ import annotations

import json
from datetime import date

import pytest

from trading_agent.alpha.market_state_classifier import MarketStateConfig
from trading_agent.contracts import BacktestConfig
from trading_agent.experts import FactorExpert, FundamentalExpert, NewsExpert, TechnicalExpert
from trading_agent.memory.expert_weights import GLOBAL_MARKET_STATE, ExpertWeightEntry


class FakeBackend:
    def __init__(self, confidence: float):
        self._confidence = confidence

    def call(self, system_prompt: str, user_prompt: str) -> str:
        tickers = json.loads(user_prompt).get("tickers", [])
        return json.dumps([
            {
                "ticker": ticker,
                "direction": "long",
                "confidence": self._confidence,
                "reasoning": "holdout",
                "evidence": ["holdout static signal"],
                "risk_flags": [],
                "data_quality": "medium",
            }
            for ticker in tickers
        ])


class StubReflection:
    def __init__(self):
        self.calls = 0

    def reflect(self, episodes, created_at, participation_rate):
        self.calls += 1
        return [ExpertWeightEntry(
            entry_id=f"stub::{created_at.isoformat()}",
            expert_id="technical",
            market_state="up_stress",
            weight_delta=0.05,
            status="learned",
            created_at=created_at,
            learned_at=created_at,
        )]


class StateEchoReflection:
    def __init__(self):
        self.calls = 0

    def reflect(self, episodes, created_at, participation_rate):
        self.calls += 1
        state = episodes[0].market_state
        return [ExpertWeightEntry(
            entry_id=f"learned::{state}::{self.calls}",
            expert_id="technical",
            market_state=state,
            weight_delta=0.05,
            status="learned",
            created_at=created_at,
            learned_at=created_at,
        )]


def _experts():
    return [
        TechnicalExpert(FakeBackend(0.9)),
        FactorExpert(FakeBackend(0.6)),
        NewsExpert(FakeBackend(0.5)),
        FundamentalExpert(FakeBackend(0.4)),
    ]


def test_expert_weight_holdout_runs_three_arms_and_freezes_memory(regime_dataset):
    from trading_agent.experiments.expert_weight_holdout import run_expert_weight_holdout

    cfg = BacktestConfig(
        date_from=date(2022, 1, 3),
        date_to=date(2022, 7, 8),
        initial_cash=100_000.0,
        commission_per_share=0.0049,
        min_commission=0.99,
        max_commission_rate=0.01,
        slippage_perc=0.0,
        liquidity_cap_pct=0.0,
        seed=11,
    )
    reflection = StubReflection()

    result = run_expert_weight_holdout(
        regime_dataset,
        cfg,
        _experts(),
        validation_boundary=date(2022, 3, 1),
        boundary=date(2022, 4, 1),
        embargo_days=2,
        market_state_config=MarketStateConfig(vol_window=5, percentile_window=20, trend_window=10, trend_confirm_days=2),
        reflection=reflection,
        reflect_every=10,
        decision_threshold_grid=(0.3, 0.4, 0.5),
    )

    assert reflection.calls > 0
    assert result.selected_decision_threshold in {0.3, 0.4, 0.5}
    assert set(result.validation_threshold_sweep) == {"0.3000", "0.4000", "0.5000"}
    assert result.validation_threshold_sweep[f"{result.selected_decision_threshold:.4f}"]["selected"] is True
    assert result.freeze_ok
    assert result.train_result.n_learned > 0
    assert result.train_result.n_applied == result.train_result.n_proposals
    assert result.reflection_trial_audit["applied_count"] == result.train_result.n_applied
    assert result.reflection_trial_audit["round_count"] == result.train_result.n_reflection_rounds
    assert result.expert_error_by_state
    assert result.weight_adjustments_by_state
    assert result.expert_adjustment_audit
    for row in result.expert_adjustment_audit:
        assert set(row) >= {
            "market_state",
            "expert_id",
            "weight_delta",
            "train_error_n",
            "train_accuracy",
            "train_mean_error_score",
            "state_signal_component_delta",
        }
    assert set(result.expert_ablation_by_state) == {"technical", "news", "fundamental"}
    for removed_expert, states in result.expert_ablation_by_state.items():
        assert states
        for state, metrics in states.items():
            assert set(metrics) >= {
                "full_mean_signal_component",
                "ablated_mean_signal_component",
                "signal_component_drop",
                "full_participation_rate",
                "ablated_participation_rate",
            }
    assert "train" in result.market_state_coverage
    assert "arms" in result.market_state_coverage
    assert "market_state_memory" in result.per_state_arm_metrics
    for arm_metrics in result.per_state_arm_metrics.values():
        assert arm_metrics
        for state_metrics in arm_metrics.values():
            assert set(state_metrics) >= {
                "episode_count",
                "mean_market_return",
                "mean_signal_component",
                "participation_rate",
            }
    assert set(result.arm_comparison_vs_equal_weight) == {
        "global_memory", "market_state_memory", "market_state_memory_online",
    }
    for comparison in result.arm_comparison_vs_equal_weight.values():
        assert set(comparison) >= {
            "total_return_delta",
            "sharpe_delta",
            "max_drawdown_delta",
            "turnover_delta",
            "participation_rate_delta",
        }
    assert set(result.research_claims) >= {
        "reflection_contribution",
        "regime_conditioning_contribution",
        "best_expert_ablation",
        "frozen_test_protocol",
    }
    assert result.system_completion["research_system_complete"] is True
    assert result.system_completion["external_paper_validation_required"] is False
    assert result.system_completion["ready_for_defense_sections"] == [
        "data_observation",
        "market_state_classifier",
        "four_experts",
        "expert_error_analysis",
        "reflection_memory",
        "train_freeze_validation_test",
        "three_arm_evaluation",
        "per_state_ablation",
    ]
    assert result.system_completion["checks"]["freeze_ok"] is True
    assert result.system_completion["checks"]["test_reflection_disabled"] is True
    assert result.system_completion["checks"]["memory_snapshot_frozen"] is True
    assert result.system_completion["checks"]["all_required_arms_present"] is True
    assert result.system_completion["checks"]["expert_error_reported"] is True
    assert result.system_completion["checks"]["per_state_ablation_reported"] is True
    assert result.research_claims["reflection_contribution"]["comparison"] == (
        "market_state_memory_vs_equal_weight"
    )
    assert result.research_claims["regime_conditioning_contribution"]["comparison"] == (
        "market_state_memory_vs_global_memory"
    )
    assert set(result.research_claims["best_expert_ablation"]) >= {
        "expert_id",
        "market_state",
        "signal_component_drop",
    }
    assert result.research_claims["frozen_test_protocol"] == {
        "freeze_ok": True,
        "allow_reflection": False,
        "memory_status": "frozen",
    }
    assert result.trial_count == 3
    assert result.configuration_trial_count == 3
    assert result.candidate_audit_count == (
        result.train_result.n_proposals + result.global_train_result.n_proposals
    )
    assert set(result.arms) == {
        "equal_weight", "global_memory", "market_state_memory",
        "market_state_memory_online",
    }
    for arm in result.arms.values():
        assert set(arm.metrics) >= {
            "total_return",
            "sharpe",
            "max_drawdown",
            "turnover",
            "participation_rate",
            "trial_count",
            "learned_adjustment_count",
            "sharpe_trial_count",
            "deflated_sharpe_ratio",
        }
        assert arm.metrics["sharpe_trial_count"] == result.trial_count
        assert arm.metrics["sharpe_trial_count"] == result.configuration_trial_count
        assert arm.loop_result.decision_threshold == result.selected_decision_threshold
    assert result.train_result.confidence_diagnostics["technical"]["n"] > 0
    assert sum(result.train_result.confidence_diagnostics["technical"]["histogram"].values()) == (
        result.train_result.confidence_diagnostics["technical"]["n"]
    )


def test_online_arm_is_causal_and_self_improves(regime_dataset):
    # The online (self-improving) arm keeps updating memory through the test window.
    # Causality guard: the online update at day t uses only data <= t (run_expert_loop
    # retrieves weights, decides, THEN observes the next return), so it cannot influence
    # the day it is computed. We verify this at the boundary: on the FIRST test date the
    # online arm and the frozen arm hold identical memory (no in-test update has run yet),
    # so their first decision must match; divergence only appears after the first in-test
    # reflection.
    from trading_agent.experiments.expert_weight_holdout import run_expert_weight_holdout

    cfg = BacktestConfig(
        date_from=date(2022, 1, 3),
        date_to=date(2022, 7, 8),
        initial_cash=100_000.0,
        commission_per_share=0.0049,
        min_commission=0.99,
        max_commission_rate=0.01,
        slippage_perc=0.0,
        liquidity_cap_pct=0.0,
        seed=11,
    )
    result = run_expert_weight_holdout(
        regime_dataset,
        cfg,
        _experts(),
        validation_boundary=date(2022, 3, 1),
        boundary=date(2022, 4, 1),
        embargo_days=2,
        market_state_config=MarketStateConfig(vol_window=5, percentile_window=20, trend_window=10, trend_confirm_days=2),
        reflection=StubReflection(),
        reflect_every=10,
        decision_threshold_grid=(0.3, 0.4, 0.5),
    )

    assert "market_state_memory_online" in result.arms
    online = result.arms["market_state_memory_online"].loop_result
    frozen = result.arms["market_state_memory"].loop_result

    # Same test window for both arms.
    assert online.n_dates == frozen.n_dates

    # Causality at the boundary: the first test-date decision is identical (the online
    # arm has not updated yet), proving day t cannot see its own outcome.
    online_states = dict(online.market_states)
    frozen_states = dict(frozen.market_states)
    first_date = min(online_states)
    assert online_states[first_date] == frozen_states[first_date]

    # The online arm reflected during the test window (self-improvement happened) and its
    # frozen counterpart did not.
    assert online.n_reflection_rounds > 0
    assert frozen.n_reflection_rounds == 0

    # The frozen train/validation guarantees are untouched by the online arm.
    assert result.freeze_ok is True
    assert result.system_completion["checks"]["test_reflection_disabled"] is True
    assert result.system_completion["checks"]["all_required_arms_present"] is True


def test_global_memory_arm_is_learned_under_one_collapsed_state(regime_dataset):
    from trading_agent.experiments.expert_weight_holdout import run_expert_weight_holdout

    cfg = BacktestConfig(
        date_from=date(2022, 1, 3),
        date_to=date(2022, 7, 8),
        initial_cash=100_000.0,
        commission_per_share=0.0049,
        min_commission=0.99,
        max_commission_rate=0.01,
        slippage_perc=0.0,
        liquidity_cap_pct=0.0,
        seed=11,
    )
    reflection = StateEchoReflection()

    result = run_expert_weight_holdout(
        regime_dataset,
        cfg,
        _experts(),
        validation_boundary=date(2022, 3, 1),
        boundary=date(2022, 4, 1),
        embargo_days=2,
        market_state_config=MarketStateConfig(vol_window=5, percentile_window=20, trend_window=10, trend_confirm_days=2),
        reflection=reflection,
        reflect_every=10,
        reflection_min_support=1,
        decision_threshold_grid=(0.3, 0.4, 0.5),
    )

    global_cards = result.arm_memory_snapshots["global_memory"]["cards"]
    state_cards = result.arm_memory_snapshots["market_state_memory"]["cards"]

    assert global_cards
    assert {c["market_state"] for c in global_cards} == {GLOBAL_MARKET_STATE}
    assert {c["market_state"] for c in state_cards} != {GLOBAL_MARKET_STATE}
    assert GLOBAL_MARKET_STATE not in {c["market_state"] for c in state_cards}


def test_attribution_mode_reaches_the_frozen_test_arms(regime_dataset):
    """Regression: attribution_mode must be threaded to every internal run, not
    just training. If the test arms silently used the default, raw and
    market_adjusted would produce identical test-arm scores."""
    from trading_agent.experiments.expert_weight_holdout import run_expert_weight_holdout

    cfg = BacktestConfig(
        date_from=date(2022, 1, 3), date_to=date(2022, 7, 8),
        initial_cash=100_000.0, seed=11,
    )

    def run(mode):
        return run_expert_weight_holdout(
            regime_dataset, cfg, _experts(),
            validation_boundary=date(2022, 3, 1), boundary=date(2022, 4, 1),
            embargo_days=2,
            market_state_config=MarketStateConfig(vol_window=5, percentile_window=20, trend_window=10, trend_confirm_days=2),
            reflection=StubReflection(), reflect_every=10,
            decision_threshold_grid=(0.3, 0.5), attribution_mode=mode,
        )

    def test_arm_scored_returns(result):
        episodes = result.arms["market_state_memory"].loop_result.episodes
        return sorted(
            round(rec.realized_next_return, 8)
            for ep in episodes for rec in ep.expert_errors
        )

    raw = test_arm_scored_returns(run("raw"))
    adjusted = test_arm_scored_returns(run("market_adjusted"))
    assert raw, "expected scored views in the frozen test arm"
    assert raw != adjusted


def test_expert_weight_holdout_supports_read_only_validation_window(regime_dataset):
    from trading_agent.experiments.expert_weight_holdout import run_expert_weight_holdout

    cfg = BacktestConfig(
        date_from=date(2022, 1, 3),
        date_to=date(2022, 7, 8),
        initial_cash=100_000.0,
        commission_per_share=0.0049,
        min_commission=0.99,
        max_commission_rate=0.01,
        slippage_perc=0.0,
        liquidity_cap_pct=0.0,
        seed=11,
    )
    reflection = StubReflection()

    result = run_expert_weight_holdout(
        regime_dataset,
        cfg,
        _experts(),
        validation_boundary=date(2022, 3, 1),
        boundary=date(2022, 5, 2),
        embargo_days=2,
        market_state_config=MarketStateConfig(vol_window=5, percentile_window=20, trend_window=10, trend_confirm_days=2),
        reflection=reflection,
        reflect_every=10,
    )

    assert set(result.validation_arms) == {"equal_weight", "global_memory", "market_state_memory"}
    assert set(result.validation_arm_comparison_vs_equal_weight) == {
        "global_memory",
        "market_state_memory",
    }
    for comparison in result.validation_arm_comparison_vs_equal_weight.values():
        assert set(comparison) >= {
            "total_return_delta",
            "sharpe_delta",
            "max_drawdown_delta",
            "turnover_delta",
            "participation_rate_delta",
        }
    assert result.split_protocol["train_end"] == "2022-03-01"
    assert result.split_protocol["validation_end"] == "2022-05-02"
    assert result.split_protocol["test_start"] > result.split_protocol["validation_end"]
    # The online (self-improving) arm reflects during the test window too, so the
    # total reflection calls include its test-window rounds on top of the two
    # training passes.
    online_rounds = result.arms["market_state_memory_online"].loop_result.n_reflection_rounds
    assert online_rounds > 0                                   # the online arm actually learned
    assert reflection.calls == (
        result.train_result.n_reflection_rounds
        + result.global_train_result.n_reflection_rounds
        + online_rounds
    )
    for arm in result.validation_arms.values():
        assert arm.loop_result.n_reflection_rounds == 0
        assert arm.loop_result.reflection_trial_records == []
    # The FROZEN test arms must not reflect; only the online arm does.
    for name, arm in result.arms.items():
        if name == "market_state_memory_online":
            continue
        assert arm.loop_result.n_reflection_rounds == 0


def test_holdout_refuses_mismatched_arm_cost_configs():
    from trading_agent.experiments.expert_weight_holdout import assert_arm_cost_configs_match

    with pytest.raises(ValueError, match="cost configuration"):
        assert_arm_cost_configs_match({
            "equal_weight": {"execution_cost_bps_per_side": 10.0},
            "market_state_memory": {"execution_cost_bps_per_side": 25.0},
        })


def test_cost_config_reports_which_cost_model_is_charged():
    from trading_agent.experiments.expert_weight_holdout import _cost_config

    base = dict(
        date_from=date(2022, 1, 3),
        date_to=date(2022, 7, 8),
        commission_per_share=0.0049,
        min_commission=0.99,
    )
    bps_cfg = BacktestConfig(**base, execution_cost_bps_per_side=10.0)
    commission_cfg = BacktestConfig(**base, execution_cost_bps_per_side=0.0)

    assert _cost_config(bps_cfg)["cost_model"] == "bps_per_side_replaces_commission"
    assert _cost_config(commission_cfg)["cost_model"] == "per_share_commission"


def test_metrics_turnover_is_traded_notional_over_mean_equity():
    from types import SimpleNamespace

    from trading_agent.experiments.expert_weight_holdout import _metrics

    equity_curve = [
        (date(2022, 1, 3), 100_000.0),
        (date(2022, 1, 4), 100_500.0),
        (date(2022, 1, 5), 99_500.0),
        (date(2022, 1, 6), 100_250.0),
        (date(2022, 1, 7), 100_100.0),
    ]
    fills = [
        SimpleNamespace(quantity=10, fill_price=100.0),
        SimpleNamespace(quantity=5, fill_price=50.0),
    ]
    result = SimpleNamespace(
        backtest=SimpleNamespace(equity_curve=equity_curve, fills=fills),
        participation_rate=0.5,
        n_learned=0,
    )

    metrics = _metrics(result, trial_count=5)

    traded_notional = 10 * 100.0 + 5 * 50.0
    mean_equity = sum(e for _, e in equity_curve) / len(equity_curve)
    assert metrics["traded_notional"] == pytest.approx(traded_notional)
    assert metrics["turnover"] == pytest.approx(traded_notional / mean_equity)
    assert metrics["n_fills"] == 2.0


def test_per_state_metrics_exclude_flat_counterfactual_rows_from_means():
    from types import SimpleNamespace

    from trading_agent.experiments.expert_weight_holdout import _per_state_metrics
    from trading_agent.memory.expert_reflection import ATTRIBUTION_COMPONENTS, AttributedEpisode

    def _episode(day, action, market, signal):
        attribution = {c: 0.0 for c in ATTRIBUTION_COMPONENTS}
        attribution["market"] = market
        attribution["signal"] = signal
        return AttributedEpisode(
            decision_date=date(2022, 1, day),
            market_state="up_stress",
            attribution=attribution,
            signal_attribution={},
            action=action,
            ticker="T1",
        )

    episodes = [
        _episode(3, "long", 0.02, 0.01),
        _episode(4, "flat", 0.0, 0.0),
        _episode(5, "flat", 0.0, 0.0),
        _episode(6, "long", -0.01, -0.005),
    ]
    out = _per_state_metrics(SimpleNamespace(episodes=episodes))["up_stress"]

    assert out["episode_count"] == 4.0
    assert out["long_episode_count"] == 2.0
    # Means are over long-action episodes only; flat counterfactual rows do not
    # dilute the components.
    assert out["mean_market_return"] == pytest.approx((0.02 - 0.01) / 2)
    assert out["mean_signal_component"] == pytest.approx((0.01 - 0.005) / 2)
    assert out["participation_rate"] == pytest.approx(0.5)


class GlobalLeakReflection:
    """Backend that always emits one per-state and one 'global' proposal."""

    def __init__(self):
        self.calls = 0

    def reflect(self, episodes, created_at, participation_rate):
        self.calls += 1
        state = episodes[0].market_state
        return [
            ExpertWeightEntry(
                entry_id=f"leak::{state}::{self.calls}",
                expert_id="technical",
                market_state=state,
                weight_delta=0.05,
                status="learned",
                created_at=created_at,
                learned_at=created_at,
            ),
            ExpertWeightEntry(
                entry_id=f"leak::global::{self.calls}",
                expert_id="news",
                market_state=GLOBAL_MARKET_STATE,
                weight_delta=0.05,
                status="learned",
                created_at=created_at,
                learned_at=created_at,
            ),
        ]


def test_per_state_training_run_drops_global_proposals(regime_dataset):
    from trading_agent.experiments.expert_weight_holdout import run_expert_weight_holdout

    cfg = BacktestConfig(
        date_from=date(2022, 1, 3),
        date_to=date(2022, 7, 8),
        initial_cash=100_000.0,
        seed=11,
    )

    result = run_expert_weight_holdout(
        regime_dataset,
        cfg,
        _experts(),
        validation_boundary=date(2022, 3, 1),
        boundary=date(2022, 4, 1),
        embargo_days=2,
        market_state_config=MarketStateConfig(vol_window=5, percentile_window=20, trend_window=10, trend_confirm_days=2),
        reflection=GlobalLeakReflection(),
        reflect_every=10,
        reflection_min_support=1,
        decision_threshold_grid=(0.3, 0.5),
    )

    state_cards = result.arm_memory_snapshots["market_state_memory"]["cards"]
    global_cards = result.arm_memory_snapshots["global_memory"]["cards"]
    assert state_cards
    assert all(c["market_state"] != GLOBAL_MARKET_STATE for c in state_cards)
    assert global_cards
    assert all(c["market_state"] == GLOBAL_MARKET_STATE for c in global_cards)


def test_temperature_grid_is_swept_selected_and_counted_as_trials(regime_dataset):
    from trading_agent.experiments.expert_weight_holdout import run_expert_weight_holdout

    cfg = BacktestConfig(
        date_from=date(2022, 1, 3), date_to=date(2022, 7, 8),
        initial_cash=100_000.0, seed=11,
    )
    result = run_expert_weight_holdout(
        regime_dataset, cfg, _experts(),
        validation_boundary=date(2022, 4, 8), boundary=date(2022, 5, 6), embargo_days=2,
        market_state_config=MarketStateConfig(vol_window=5, percentile_window=20, trend_window=10, trend_confirm_days=2),
        reflect_every=10, reflection_min_support=1,
        decision_threshold_grid=(0.3, 0.5),
        temperature_grid=(0.0005, 0.001, 0.002),
    )
    # The selected temperature is one of the grid points, and the deflation trial count is
    # temperatures x thresholds (the temperature sweep is a pure numeric re-read, but it is
    # still a configuration choice and must be paid for in the deflated Sharpe).
    assert result.selected_temperature in (0.0005, 0.001, 0.002)
    assert result.trial_count == 3 * 2
    assert result.split_protocol["temperature_grid"] == [0.0005, 0.001, 0.002]
    assert result.split_protocol["selected_temperature"] == result.selected_temperature
