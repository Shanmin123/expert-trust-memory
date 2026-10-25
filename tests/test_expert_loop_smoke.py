"""Smoke tests for the expert-reflection loop."""

from __future__ import annotations

import json
import threading
from datetime import date

import pandas as pd

from trading_agent.alpha.alpha158_recipes import REGISTERED_ALPHA158_FEATURES
from trading_agent.alpha.alpha158_strategy import Alpha158StrategyAlpha, FrameAlpha158FeatureProvider
from trading_agent.experts import (
    FactorExpert,
    FundamentalExpert,
    NewsExpert,
    TechnicalExpert,
)
from trading_agent.alpha.market_state_classifier import MARKET_STATES, MarketStateConfig
from trading_agent.data.finsaber_news import FinsaberNewsPanel, NewsItem
from trading_agent.expert_context import ExpertSignalSources
from trading_agent.memory.expert_weights import ExpertWeightEntry, ExpertWeightStore
from trading_agent.runner import ExpertLoopResult, _confidence_summary, run_expert_loop


class FakeBackend:
    """Returns a canned ExpertView JSON with a per-expert confidence."""

    def __init__(self, confidence: float):
        self._conf = confidence
        self.calls: list[tuple[str, str]] = []

    def call(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        payload = json.loads(user_prompt)
        tickers = payload.get("tickers", [])
        return json.dumps([
            {
                "ticker": ticker,
                "direction": "long",
                "confidence": self._conf,
                "reasoning": "smoke",
                "evidence": ["static smoke signal"],
                "risk_flags": [],
                "data_quality": "medium",
            }
            for ticker in tickers
        ])


def _experts():
    return [
        TechnicalExpert(FakeBackend(0.9)),
        FactorExpert(FakeBackend(0.6)),
        NewsExpert(FakeBackend(0.5)),
        FundamentalExpert(FakeBackend(0.4)),
    ]


_FAST_MSC = MarketStateConfig(vol_window=5, percentile_window=20, trend_window=10, trend_confirm_days=2)


def test_expert_loop_runs_end_to_end(regime_dataset, mem_backtest_config):
    result = run_expert_loop(
        regime_dataset,
        mem_backtest_config,
        _experts(),
        market_state_config=_FAST_MSC,
        reflect_every=10,
    )

    assert isinstance(result, ExpertLoopResult)
    assert result.n_dates > 0
    assert result.n_active_dates > 0
    assert result.backtest is not None
    assert len(result.backtest.equity_curve) > 0
    assert len(result.market_states) == result.n_dates
    allowed = set(MARKET_STATES) | {"warmup"}
    assert all(s in allowed for _, s in result.market_states)
    assert set(s for _, s in result.market_states) & set(MARKET_STATES)
    assert all(ep.market_state in MARKET_STATES for ep in result.episodes)
    assert all(ep.market_state in MARKET_STATES for ep in result.episodes)
    assert result.n_reflection_rounds > 0
    assert 0.0 <= result.participation_rate <= 1.0
    assert len(result.episodes) >= result.n_active_dates
    assert {ep.action for ep in result.episodes} <= {"long", "flat"}
    assert all(ep.ticker is not None for ep in result.episodes)


def test_attribution_default_is_raw_and_toggles_to_market_adjusted(
    regime_dataset, mem_backtest_config
):
    def run(mode=None):
        kwargs = dict(market_state_config=_FAST_MSC, allow_reflection=False)
        if mode is not None:
            kwargs["attribution_mode"] = mode
        return run_expert_loop(regime_dataset, mem_backtest_config, _experts(), **kwargs)

    # Expert error records score every non-default view (regardless of the
    # aggregate action) on the return; that return is the raw per-name return
    # in raw mode and the excess over the market proxy in market_adjusted.
    def scored_returns(result):
        return sorted(
            round(rec.realized_next_return, 8)
            for ep in result.episodes for rec in ep.expert_errors
        )

    raw = scored_returns(run("raw"))
    adjusted = scored_returns(run("market_adjusted"))
    default = scored_returns(run())

    assert raw, "expected non-default expert views to score"
    assert raw != adjusted            # market adjustment shifts the scores
    assert default == raw             # default is the not-demeaned counterfactual (raw)


def test_first_decision_uses_equal_weights_causal_scorecard_ordering(
    regime_dataset, mem_backtest_config
):
    # CAUSAL GUARD: the scorecard is observed only AFTER the decision, execution, and
    # attribution, so the weights used on any date depend only on earlier outcomes. With
    # min_support=1 the store would tilt off equal weights immediately if observe ran before
    # get_weights; asserting the FIRST decision's weights are equal one-vote-vetoes that
    # look-ahead ordering bug (the first active date has an empty scorecard).
    from trading_agent.memory.expert_scorecard import ExpertScorecardStore, equal_weights

    class RecordingStore(ExpertScorecardStore):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.returned = []

        def get_weights(self, market_state):
            w = super().get_weights(market_state)
            self.returned.append(dict(w))
            return w

    store = RecordingStore(min_support=1)      # tilts after a single observation
    run_expert_loop(
        regime_dataset, mem_backtest_config, _experts(),
        market_state_config=_FAST_MSC, weight_store=store,
        reflect_every=5, allow_reflection=True,
    )
    assert store.returned, "expected at least one decision"
    assert store.returned[0] == equal_weights()   # first decision saw an empty scorecard
    # and the scorecard does eventually tilt, so the guard above is discriminating.
    assert any(w != equal_weights() for w in store.returned)


def test_trained_memory_is_invariant_across_the_tuning_grid(regime_dataset, mem_backtest_config):
    # GRID-INVARIANCE: the scorecard (counterfactual EW moments) and the reflection delta
    # must not depend on the tuned temperature or decision threshold. That independence is
    # what lets the whole tuning grid be replayed over a single trained memory instead of
    # retraining per grid point. Train twice with very different (tau, theta); the stored
    # scorecard cards -- EW moments AND reflection deltas -- must be byte-identical. If
    # anyone ever feeds a tuned-parameter-dependent quantity (participation, a composed
    # score, portfolio P&L) into reflection, this test fails immediately.
    from trading_agent.memory.expert_scorecard import ExpertScorecardStore

    def train(tau, theta):
        store = ExpertScorecardStore(temperature=tau, min_support=1)
        run_expert_loop(
            regime_dataset, mem_backtest_config, _experts(),
            market_state_config=_FAST_MSC, weight_store=store,
            reflect_every=5, reflection_min_support=1,
            decision_threshold=theta, allow_reflection=True,
        )
        return store.snapshot()["cards"]

    a = train(0.0005, 0.2)
    b = train(0.0020, 0.7)
    assert a, "expected the fixture to train a non-empty scorecard"
    assert a == b       # EW moments + reflection deltas identical across (tau, theta)


def test_decision_cadence_carries_positions_but_leaves_memory_identical(
    regime_dataset, mem_backtest_config
):
    # Coarser decision cadence re-votes only every Nth date; between votes positions carry.
    # Learning evidence is action-independent, so the trained memory must be byte-identical
    # to daily cadence (same guard family as grid-invariance).
    from trading_agent.memory.expert_scorecard import ExpertScorecardStore

    def train(cadence):
        store = ExpertScorecardStore(min_support=1)
        result = run_expert_loop(
            regime_dataset, mem_backtest_config, _experts(),
            market_state_config=_FAST_MSC, weight_store=store,
            reflect_every=5, reflection_min_support=1,
            decision_threshold=0.0, allow_reflection=True,
            decision_cadence=cadence,
        )
        return store.snapshot()["cards"], result

    cards_daily, res_daily = train(1)
    cards_weekly, res_weekly = train(5)
    assert cards_daily == cards_weekly          # memory unchanged by the cadence
    assert cards_daily                          # and non-empty

    # With always-long experts the daily loop is long from day one; the weekly loop
    # must reach the same steady state (carry preserves the vote made on decision dates).
    assert res_weekly.participation_rate > 0

    import pytest
    with pytest.raises(ValueError, match="decision_cadence"):
        run_expert_loop(
            regime_dataset, mem_backtest_config, _experts(),
            market_state_config=_FAST_MSC, allow_reflection=False,
            decision_cadence=0,
        )


def test_expert_loop_rejects_unknown_attribution_mode(regime_dataset, mem_backtest_config):
    import pytest
    with pytest.raises(ValueError, match="attribution_mode"):
        run_expert_loop(
            regime_dataset, mem_backtest_config, _experts(),
            market_state_config=_FAST_MSC, allow_reflection=False,
            attribution_mode="nonsense",
        )


def test_confidence_diagnostics_split_real_parse_and_data_unavailable_judgments():
    diagnostics = _confidence_summary({
        "technical": [
            {
                "confidence": 0.7,
                "direction": "long",
                "safe_default": False,
                "safe_default_reason": None,
            },
            {
                "confidence": 0.0,
                "direction": "flat",
                "safe_default": True,
                "safe_default_reason": "parse_error",
            },
            {
                "confidence": 0.0,
                "direction": "flat",
                "safe_default": True,
                "safe_default_reason": "data_unavailable",
            },
        ],
    })

    technical = diagnostics["technical"]
    assert technical["n"] == 3
    assert technical["real_judgment_count"] == 1
    assert technical["safe_default_count"] == 2
    assert technical["safe_default_reasons"] == {
        "data_unavailable": 1,
        "parse_error": 1,
        "unknown": 0,
    }
    assert technical["real_judgment_rate"] == 1 / 3


class _StubReflection:
    """Always proposes one learned adjustment per round."""

    def __init__(self):
        self.calls = 0

    def reflect(self, episodes, created_at, participation_rate):
        self.calls += 1
        return [ExpertWeightEntry(
            entry_id=f"stub::technical::normal::{created_at.isoformat()}",
            expert_id="technical",
            market_state="up_stress",
            weight_delta=0.05,
            status="learned",
            created_at=created_at,
            learned_at=created_at,
        )]


def test_expert_loop_exercises_reflection_and_learning(regime_dataset, mem_backtest_config):
    stub = _StubReflection()
    result = run_expert_loop(
        regime_dataset,
        mem_backtest_config,
        _experts(),
        market_state_config=_FAST_MSC,
        reflection=stub,
        reflect_every=10,
    )
    assert stub.calls > 0
    assert result.n_proposals > 0
    assert result.n_applied == result.n_proposals
    assert result.n_learned > 0


class _ZeroDeltaReflection:
    def reflect(self, episodes, created_at, participation_rate):
        return [ExpertWeightEntry(
            entry_id=f"zero::technical::normal::{created_at.isoformat()}",
            expert_id="technical",
            market_state="up_stress",
            weight_delta=0.0,
            status="learned",
            created_at=created_at,
            learned_at=created_at,
        )]


def test_expert_loop_writes_valid_proposals_directly_to_learned_memory(
    regime_dataset, mem_backtest_config
):
    store = ExpertWeightStore()
    result = run_expert_loop(
        regime_dataset,
        mem_backtest_config,
        _experts(),
        market_state_config=_FAST_MSC,
        weight_store=store,
        reflection=_ZeroDeltaReflection(),
        reflect_every=10,
    )

    assert result.n_proposals > 0
    assert result.n_applied == result.n_proposals
    assert result.n_learned == result.n_proposals
    assert store.count_learned() == result.n_proposals
    assert all(round_record["n_applied"] == round_record["n_proposals"]
               for round_record in result.reflection_trial_records)


def test_expert_loop_reflects_on_recent_window_only(regime_dataset, mem_backtest_config):
    """Cadence and window are measured in active episode DATES, not episode counts."""

    class CaptureReflection:
        def __init__(self):
            self.window_date_counts: list[int] = []
            self.window_lengths: list[int] = []

        def reflect(self, episodes, created_at, participation_rate):
            self.window_date_counts.append(len({ep.decision_date for ep in episodes}))
            self.window_lengths.append(len(episodes))
            return []

    capture = CaptureReflection()
    result = run_expert_loop(
        regime_dataset,
        mem_backtest_config,
        _experts(),
        market_state_config=_FAST_MSC,
        reflection=capture,
        reflect_every=5,
        reflection_window=7,
        reflection_min_support=5,
    )

    n_tickers = 4  # regime_dataset universe
    assert result.n_reflection_rounds > 0
    assert capture.window_date_counts
    # The window spans at most 7 distinct episode dates (per-name episodes within).
    assert max(capture.window_date_counts) <= 7
    assert any(count == 7 for count in capture.window_date_counts)
    assert max(capture.window_lengths) <= 7 * n_tickers
    # With the full universe active, per-name episodes exceed the date window,
    # which is exactly what the old per-episode window got wrong.
    assert max(capture.window_lengths) > 7


def test_expert_loop_reflection_cadence_counts_dates_not_per_name_episodes(
    regime_dataset, mem_backtest_config
):
    class CountReflection:
        def __init__(self):
            self.calls = 0

        def reflect(self, episodes, created_at, participation_rate):
            self.calls += 1
            return []

    counter = CountReflection()
    result = run_expert_loop(
        regime_dataset,
        mem_backtest_config,
        _experts(),
        market_state_config=_FAST_MSC,
        reflection=counter,
        reflect_every=10,
        reflection_window=63,
    )

    episode_dates = {ep.decision_date for ep in result.episodes}
    # One reflection per 10 active episode dates: with a 4-ticker universe the old
    # per-episode trigger would have fired ~4x as often.
    assert counter.calls == len(episode_dates) // 10


def test_expert_loop_never_reads_weights_after_same_date_reflection_write(
    regime_dataset, mem_backtest_config
):
    """Leakage ordering: weights for date t must be read before reflection at t writes.

    The episode labels consumed by reflection at date t use the t+1 price, so a
    proposal applied to the same date's weights would be look-ahead. The loop must
    read get_weights before add_learned_adjustment on every date.
    """

    events: list[tuple[str, object]] = []

    class OrderAuditStore(ExpertWeightStore):
        def get_weights(self, market_state, *, include_global=True):
            events.append(("read", None))
            return super().get_weights(market_state, include_global=include_global)

        def add_learned_adjustment(self, entry):
            events.append(("write", None))
            return super().add_learned_adjustment(entry)

    def on_progress(event):
        if event["event"] == "date_start":
            events.append(("date", event["date"]))

    run_expert_loop(
        regime_dataset,
        mem_backtest_config,
        _experts(),
        market_state_config=_FAST_MSC,
        weight_store=OrderAuditStore(),
        reflection=_StubReflection(),
        reflect_every=1,
        progress_callback=on_progress,
    )

    writes_seen = 0
    saw_write_this_date = False
    for kind, _ in events:
        if kind == "date":
            saw_write_this_date = False
        elif kind == "write":
            saw_write_this_date = True
            writes_seen += 1
        elif kind == "read":
            assert not saw_write_this_date, (
                "store weights were read after a same-date reflection write"
            )
    assert writes_seen > 0


def test_expert_loop_reuses_preseeded_learned_memory(regime_dataset, mem_backtest_config):
    store = ExpertWeightStore()
    seed = ExpertWeightEntry(
        entry_id="seed::news::normal",
        expert_id="news",
        market_state="up_stress",
        weight_delta=0.10,
        status="learned",
        created_at=date(2020, 1, 1),
        learned_at=date(2020, 2, 1),
    )
    store.add_learned_adjustment(seed)
    assert store.count_learned() == 1

    result = run_expert_loop(
        regime_dataset,
        mem_backtest_config,
        _experts(),
        market_state_config=_FAST_MSC,
        weight_store=store,
        reflect_every=10,
    )
    w = store.get_weights("up_stress")
    assert w["news"] > 1.0 / 3.0
    assert result.n_dates > 0
    assert result.n_learned >= 1


def test_expert_loop_does_not_feed_learned_rationales_as_prior_lessons(
    regime_dataset, mem_backtest_config
):
    store = ExpertWeightStore()
    seed = ExpertWeightEntry(
        entry_id="seed::technical::global",
        expert_id="technical",
        market_state="global",
        weight_delta=0.05,
        status="learned",
        created_at=date(2020, 1, 1),
        learned_at=date(2020, 2, 1),
        rationale="trend evidence helped in similar windows",
    )
    store.add_learned_adjustment(seed)
    technical_backend = FakeBackend(0.9)
    experts = [
        TechnicalExpert(technical_backend),
        FactorExpert(FakeBackend(0.6)),
        NewsExpert(FakeBackend(0.5)),
        FundamentalExpert(FakeBackend(0.4)),
    ]

    run_expert_loop(
        regime_dataset,
        mem_backtest_config,
        experts,
        market_state_config=_FAST_MSC,
        weight_store=store,
        allow_reflection=False,
    )

    payloads = [json.loads(user) for _, user in technical_backend.calls]
    assert all("prior_lessons" not in payload for payload in payloads)
    assert all("trend evidence helped" not in json.dumps(payload) for payload in payloads)


def test_expert_loop_passes_real_signal_context_blocks(regime_dataset, mem_backtest_config):
    technical_backend = FakeBackend(0.9)
    factor_backend = FakeBackend(0.6)
    news_backend = FakeBackend(0.5)
    fundamental_backend = FakeBackend(0.4)
    experts = [
        TechnicalExpert(technical_backend),
        FactorExpert(factor_backend),
        NewsExpert(news_backend),
        FundamentalExpert(fundamental_backend),
    ]

    feature = REGISTERED_ALPHA158_FEATURES[0]
    dates = list(pd.bdate_range(mem_backtest_config.date_from, mem_backtest_config.date_to).date)
    # Per-date history: the factor signal is a time-series z-score against each
    # ticker's own prior values, so a single-date frame yields no signal.
    factor_frame = pd.DataFrame(
        [
            {"date": d, "ticker": ticker, feature.strategy_id: base + 0.02 * ((i % 5) - 2)}
            for i, d in enumerate(dates)
            for ticker, base in (("T1", 1.0), ("T2", 0.5), ("T3", -0.5), ("T4", -1.0))
        ]
    )
    signal_sources = ExpertSignalSources(
        factor_alpha=Alpha158StrategyAlpha(
            FrameAlpha158FeatureProvider(factor_frame),
            features=(feature,),
            history_window=120,
            min_history=20,
        ),
        news_panel=FinsaberNewsPanel.from_items([
            NewsItem(date=dates[-1], ticker="T1", text="T1 reports strong growth"),
        ]),
    )

    run_expert_loop(
        regime_dataset,
        mem_backtest_config,
        experts,
        market_state_config=_FAST_MSC,
        signal_sources=signal_sources,
        allow_reflection=False,
    )

    tech_payloads = [json.loads(user) for _, user in technical_backend.calls]
    assert any(
        set(payload.get("technical_signals", {}).get("by_ticker", {}).get("T1", {}))
        >= {"momentum", "reversal", "lowvol"}
        for payload in tech_payloads
    )
    assert any(
        payload["technical_signals"]["by_ticker"]["T1"]["available"] is True
        for payload in tech_payloads
        if "technical_signals" in payload
    )

    factor_payloads = [json.loads(user) for _, user in factor_backend.calls]
    assert any(payload["factor_signals"]["available"] is True for payload in factor_payloads)
    assert any("T1" in payload["factor_signals"]["by_ticker"] for payload in factor_payloads)

    news_payloads = [json.loads(user) for _, user in news_backend.calls]
    assert any(payload["news_signals"]["available"] is True for payload in news_payloads)
    assert any("T1" in payload["news_signals"]["by_ticker"] for payload in news_payloads)
    # Design D (TradingAgents-style news analyst): the news expert now READS recent news
    # text — the trailing snippets must reach the prompt as `recent_news`.
    assert any("reports strong growth" in json.dumps(payload) for payload in news_payloads)

    fundamental_payloads = [json.loads(user) for _, user in fundamental_backend.calls]
    assert any(
        "market_state" not in json.dumps(payload.get("fundamental_inputs", {}))
        and "breadth_above_50d" in payload["fundamental_inputs"]
        and "realized_vol_20d" in payload["fundamental_inputs"]
        and "by_ticker" in payload["fundamental_inputs"]
        for payload in fundamental_payloads
    )


def test_expert_loop_can_run_with_reflection_disabled(regime_dataset, mem_backtest_config):
    store = ExpertWeightStore()
    before = store.snapshot()
    result = run_expert_loop(
        regime_dataset,
        mem_backtest_config,
        _experts(),
        market_state_config=_FAST_MSC,
        weight_store=store,
        allow_reflection=False,
    )
    assert result.n_reflection_rounds == 0
    assert result.n_proposals == 0
    assert result.n_applied == 0
    assert store.snapshot() == before


def test_expert_loop_emits_progress_events(regime_dataset, mem_backtest_config):
    events = []

    run_expert_loop(
        regime_dataset,
        mem_backtest_config,
        _experts(),
        market_state_config=_FAST_MSC,
        allow_reflection=False,
        progress_callback=events.append,
    )

    assert any(event["event"] == "date_start" for event in events)
    assert any(event["event"] == "expert_start" and event["expert_id"] == "technical" for event in events)
    assert any(event["event"] == "expert_done" and event["expert_id"] == "technical" for event in events)
    assert any(event["event"] == "date_done" for event in events)


def test_expert_loop_can_call_experts_in_parallel(regime_dataset, mem_backtest_config):
    barrier = threading.Barrier(4, timeout=5)

    class BlockingBackend(FakeBackend):
        def call(self, system_prompt: str, user_prompt: str) -> str:
            barrier.wait()
            return super().call(system_prompt, user_prompt)

    experts = [
        TechnicalExpert(BlockingBackend(0.9)),
        FactorExpert(BlockingBackend(0.6)),
        NewsExpert(BlockingBackend(0.5)),
        FundamentalExpert(BlockingBackend(0.4)),
    ]

    result = run_expert_loop(
        regime_dataset,
        mem_backtest_config,
        experts,
        market_state_config=_FAST_MSC,
        allow_reflection=False,
        parallel_expert_calls=True,
    )

    assert result.n_active_dates > 0


def test_attribution_horizon_embargoes_learning_by_h_active_bars(
    regime_dataset, mem_backtest_config, monkeypatch
):
    """LOOK-AHEAD GUARD for horizon attribution.

    The learning signal g = v * r_{t->t+h} uses the price h bars after date t, which is
    realized only at t+h. An episode from active-date index k must therefore enter the
    scorecard exactly h-1 active bars LATER than the always-safe h=1 baseline (which folds
    after date k's own decision). If any episode entered earlier, a decision between t and
    t+h would read a signal derived from its own future -- exactly the look-ahead this
    dissertation studies. The invariant `reads_at_release == k + h - 1` holds for BOTH the
    h==1 immediate path and the h>1 embargo path, so one assertion covers both.
    """
    import trading_agent.runner as runner_mod
    from trading_agent.memory.expert_scorecard import ExpertScorecardStore

    def run(h):
        active_order: list = []
        reads = [0]
        released: list[tuple[int, object]] = []

        class ReadCountingStore(ExpertScorecardStore):
            def get_weights(self, market_state):
                reads[0] += 1
                return super().get_weights(market_state)

        store = ReadCountingStore(min_support=1)
        orig = runner_mod._attribute

        def spy(decision_date, *a, **k):
            released.append((reads[0], decision_date))
            return orig(decision_date, *a, **k)

        monkeypatch.setattr(runner_mod, "_attribute", spy)

        def on_progress(ev):
            if ev["event"] == "date_done" and ev.get("active"):
                active_order.append(ev["date"])

        run_expert_loop(
            regime_dataset, mem_backtest_config, _experts(),
            market_state_config=_FAST_MSC, weight_store=store,
            reflect_every=5, reflection_min_support=1,
            attribution_horizon=h, allow_reflection=True,
            progress_callback=on_progress,
        )
        monkeypatch.setattr(runner_mod, "_attribute", orig)
        index_of = {d: i + 1 for i, d in enumerate(active_order)}
        return released, index_of, store.snapshot()["cards"]

    for h in (1, 5):
        released, index_of, _ = run(h)
        assert released, f"expected episodes to be attributed at h={h}"
        checked = 0
        for reads_at_release, d in released:
            if d not in index_of:
                continue  # tail episode never re-observed in-window; not a leak
            k = index_of[d]
            assert reads_at_release == k + h - 1, (
                f"h={h}: episode from active index {k} folded after {reads_at_release} "
                f"reads; embargo requires exactly {k + h - 1} (leak if smaller)"
            )
            checked += 1
        assert checked > 0, f"h={h}: no in-window episodes verified"

    # discriminating: the horizon actually changes what is learned.
    _, _, cards_h1 = run(1)
    _, _, cards_h5 = run(5)
    assert cards_h1 != cards_h5, "h=5 must train a different memory than h=1"
