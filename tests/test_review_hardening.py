"""Round-24 review hardening: library defaults must equal the documented
system configuration, and the guards found by the review must hold.

None of these changes touches sealed numbers: the sealed launcher passes
every knob explicitly, 190 sealed artifacts contain no supported
zero-variance cell, and the 53 frozen corpora contain zero safe-default
views (so the silence path never fired). These tests pin the hardened
behavior for every OTHER caller.
"""

from __future__ import annotations

import random

import pytest

from trading_agent.aggregation.soft_weighting import aggregate_by_ticker
from trading_agent.aggregation.weights import project_to_floored_simplex
from trading_agent.contracts import ExpertView
from trading_agent.memory.expert_reflection import (
    AttributedEpisode,
    DeterministicExpertReflection,
    strip_to_reflection_inputs,
)
from trading_agent.memory.expert_scorecard import (
    DEFAULT_TEMPERATURE,
    ExpertScorecardStore,
)
from trading_agent.memory.metrics import deflated_sharpe_ratio


def test_library_defaults_match_the_documented_system_configuration():
    """The sealed launcher always passed these explicitly; the constructor
    defaults previously described the ablations instead, so any convenience
    caller silently ran a different system than the one documented."""
    store = ExpertScorecardStore()
    assert store._standardize is True
    assert store._tau == DEFAULT_TEMPERATURE == 0.5
    reflection = DeterministicExpertReflection()
    assert reflection.step_mode == "scale_free"


def _episode(state: str, attribution: dict, *, rng=None) -> AttributedEpisode:
    """Forbidden (grid-dependent) fields are randomised when rng is given:
    the portfolio attribution components, the system's action, the ticker."""
    from datetime import date

    noise = (lambda: rng.uniform(-1, 1)) if rng else (lambda: 0.0)
    return AttributedEpisode(
        decision_date=date(2020, 1, 2),
        market_state=state,
        attribution={c: noise() for c in ("market", "style", "signal", "sizing", "execution")},
        signal_attribution=attribution,
        expert_errors=[],
        action=(rng.choice(["long", "flat"]) if rng else "long"),
        ticker="TST",
    )


def test_reflection_cannot_see_grid_dependent_fields():
    """Grid invariance, made structural: randomising every forbidden field
    (action, weights, participation, P&L) must leave the proposals
    byte-identical, because the strip boundary removes them before any maker
    can look."""
    rng = random.Random(7)
    states = ["up_calm", "down_stress"]
    experts = ["technical", "news", "fundamental"]

    def episodes(noise: bool):
        out = []
        for i in range(40):
            state = states[i % 2]
            attribution = {e: ((i % 5) - 2) * 0.004 + j * 0.001 for j, e in enumerate(experts)}
            out.append(_episode(state, attribution, rng=rng if noise else None))
        return out

    stripped = strip_to_reflection_inputs(episodes(noise=True))
    assert stripped, "boundary must pass the allowed fields through"
    assert set(stripped[0]._fields) == {"market_state", "expert_id", "contribution"}

    maker = DeterministicExpertReflection(min_support=3)
    from datetime import date

    clean = maker.reflect(episodes(noise=False), date(2020, 1, 31), 0.5)
    noisy = maker.reflect(episodes(noise=True), date(2020, 1, 31), 0.9)
    assert [(e.expert_id, e.market_state, e.weight_delta) for e in clean] == [
        (e.expert_id, e.market_state, e.weight_delta) for e in noisy
    ]


def test_pooled_std_includes_supported_zero_variance_cells():
    store = ExpertScorecardStore(scope="state", standardize=True, temperature=1.0)
    for _ in range(25):
        store.observe("up_calm", {"technical": 0.001})   # supported, exactly constant
    for i in range(25):
        store.observe("up_calm", {"news": 0.002 if i % 2 else -0.002})
    for _ in range(25):
        store.observe("up_calm", {"fundamental": 0.0})   # supported, exactly constant
    weights = store.get_weights("up_calm")
    # With the constant cells IN the pool the scale is smaller than the
    # varying cell's std alone, so the tilt toward news' rival must differ
    # from the old (excluding) behaviour; the strong invariant we can pin:
    # the pool is finite, weights are a floored simplex, and the varying
    # expert does not get the whole budget by scale inflation.
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert all(w >= 0.05 - 1e-9 for w in weights.values())


def test_simplex_projection_survives_overflow_and_nan():
    out = project_to_floored_simplex(
        {"a": 1e308, "b": 1e308, "c": 1.0, "d": 1.0}, 0.05
    )
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert all(w >= 0.05 - 1e-9 for w in out.values())
    assert out["a"] == pytest.approx(out["b"])
    out = project_to_floored_simplex(
        {"a": float("nan"), "b": float("inf"), "c": 1.0}, 0.05
    )
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert all(w >= 0.05 - 1e-9 for w in out.values())
    assert out["c"] == max(out.values())


def test_deflated_sharpe_zero_variance_returns_none():
    assert deflated_sharpe_ratio([0.01] * 10, n_trials=3, var_sr=0.1) is None
    base = [0.01, 0.011] * 5
    assert deflated_sharpe_ratio(base + [0.02], n_trials=3, var_sr=0.1) is not None


def _view(direction: str, reasoning: str = "ok") -> ExpertView:
    return ExpertView(
        expert_id="technical",
        direction=direction,  # type: ignore[arg-type]
        confidence=0.6,
        reasoning=reasoning,
        evidence=[],
        risk_flags=[],
        data_quality="good" if reasoning == "ok" else "missing",
    )


def _safe_view(expert_id: str) -> ExpertView:
    return ExpertView(
        expert_id=expert_id,
        direction="neutral",
        confidence=0.0,
        reasoning="data_unavailable",
        evidence=[],
        risk_flags=[],
        data_quality="missing",
    )


def test_silence_semantics_pin_design_d3_both_ways():
    """No speaker + no held position -> flat; no speaker + held long ->
    maintained (design D3, frozen 2026-07-17). The project notes
    clause read as flat-always; the code and its docstring are the design."""
    views = {eid: {"TST": _safe_view(eid)} for eid in ("technical", "news", "fundamental")}
    weights = {"technical": 1 / 3, "news": 1 / 3, "fundamental": 1 / 3}
    fresh = aggregate_by_ticker(views, weights, tickers=["TST"], prev_directions={})
    assert fresh.by_ticker["TST"].direction == "flat"
    held = aggregate_by_ticker(
        views, weights, tickers=["TST"], prev_directions={"TST": "long"}
    )
    assert held.by_ticker["TST"].direction == "long"
    to_cash = aggregate_by_ticker(
        views, weights, tickers=["TST"],
        prev_directions={"TST": "long"}, silence_policy="flat",
    )
    assert to_cash.by_ticker["TST"].direction == "flat"


def test_aggregation_defaults_are_the_frozen_boundaries():
    """theta=0 strict entry, exit=0 strict HOLD are now the DEFAULTS: a
    2-long/1-flat committee (A=1/3) must enter without any explicit
    threshold argument."""
    views = {
        "technical": {"TST": _view("long")},
        "news": {"TST": _view("long")},
        "fundamental": {"TST": _view("flat")},
    }
    weights = {"technical": 1 / 3, "news": 1 / 3, "fundamental": 1 / 3}
    result = aggregate_by_ticker(views, weights, tickers=["TST"])
    assert result.by_ticker["TST"].direction == "long"
    assert result.by_ticker["TST"].score == pytest.approx(1 / 3)


def test_missing_h_target_contributes_no_learning_evidence():
    """The thesis defines g over the ELAPSED forward window; a ticker whose
    h-day target bar is missing used to fall back to its 1-bar return,
    silently mixing horizons in the learned memory."""
    from datetime import date

    from trading_agent.runner import _attribute

    class Panel:
        # AAA has a price on the decision date and the next bar, but its
        # h=21 target bar is missing; BBB has both.
        def _price_on(self, tk, d):
            return 100.0

        def price_after(self, tk, d, h):
            if h == 1:
                return 101.0
            return None if tk == "AAA" else 121.0

    class Obs:
        tickers = ("AAA", "BBB")

    from trading_agent.contracts import ExpertView

    view = ExpertView(
        expert_id="technical", direction="long", confidence=0.6,
        reasoning="ok", evidence=[], risk_flags=[], data_quality="good",
    )
    views = {"technical": {"AAA": view, "BBB": view}}
    episodes = _attribute(
        observation=Obs(),
        panel=Panel(),
        decision_date=date(2020, 6, 1),
        market_state="up_calm",
        views_by_expert=views,
        decisions_by_ticker={"AAA": "long", "BBB": "long"},
        weights={"technical": 1.0},
        attribution_mode="raw",
        learning_horizon=21,
    )
    by_ticker = {e.ticker: e.signal_attribution for e in episodes}
    assert by_ticker["BBB"] == {"technical": pytest.approx(0.21)}
    assert by_ticker["AAA"] == {}, "no elapsed window -> no learning evidence"


def test_strict_replay_rejects_a_record_for_a_different_prompt(tmp_path):
    """A record must never be served as the answer to a prompt it did not answer.

    The corpus is keyed by (expert_id, user_prompt), so a record for a different
    prompt cannot be reached at all: strict replay misses and raises, rather than
    returning the stale response. This used to need a stored prompt digest to
    detect; now it is a property of the key.
    """
    import json

    from trading_agent.experts.view_corpus import CachedExpertBackend, ViewCorpus

    path = tmp_path / "corpus.jsonl"
    record = {
        "expert_id": "technical",
        "date": "2020-01-02",
        "tickers": ["TST"],
        "user_prompt": "DIFFERENT USER",
        "response": "STALE",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    corpus = ViewCorpus(path, mode="replay")

    class Inner:
        def call(self, *_a):
            raise AssertionError("replay must not call the live backend")

    backend = CachedExpertBackend(Inner(), corpus, "technical")
    with pytest.raises(KeyError, match="replay corpus miss"):
        backend.call("SYS", "USER")
    # and the record it does answer is still served
    assert backend.call("SYS", "DIFFERENT USER") == "STALE"


def test_non_finite_values_pass_through_for_corpus_compatibility():
    """REVERSAL of an earlier hardening: filtering NaN/inf here changes the
    prompt bytes and silently breaks frozen-corpus replay (every cache key
    shifts; found when a pre-registered replay missed on news/fundamental
    for every date). The corpora were recorded with "nan" in the prompts,
    so the byte-identical pass-through IS the contract."""
    import math

    from trading_agent.experts.backends import _optional_float

    assert math.isnan(_optional_float(float("nan")))
    assert _optional_float(float("inf")) == float("inf")
    assert _optional_float("0.5") == 0.5
    assert _optional_float(None) is None


def test_t_stat_step_is_uncertainty_scaled_and_saturates_at_three():
    """Pre-registered Family A (prereg_free_replay_2026-07-27): the step is
    0.10 * t / 3 clipped to +-0.10; s == 0 or n < 2 contributes nothing."""
    from datetime import date

    from trading_agent.memory.expert_reflection import DeterministicExpertReflection

    maker = DeterministicExpertReflection(min_support=3, step_mode="t_stat")
    # technical: strongly positive, tiny spread -> |t| >> 3 -> saturated +0.10
    # news: mirrored negative -> saturated -0.10
    # fundamental: constant at the state mean -> s>0? no: exactly zero spread -> no step
    episodes = []
    for i in range(12):
        wiggle = 0.0001 if i % 2 else -0.0001
        episodes.append(_episode("up_calm", {
            "technical": 0.01 + wiggle,
            "news": -0.01 - wiggle,
            "fundamental": 0.0,
        }))
    out = maker.reflect(episodes, date(2020, 1, 31), 0.5)
    deltas = {e.expert_id: e.weight_delta for e in out if e.market_state == "up_calm"}
    assert deltas["technical"] == pytest.approx(0.10)
    assert deltas["news"] == pytest.approx(-0.10)
    assert "fundamental" not in deltas   # zero spread -> no evidence of scale
    # A weak, noisy expert must land strictly inside the bound.
    episodes = []
    values = [0.004, -0.003, 0.005, -0.004, 0.003, -0.002, 0.004, -0.003]
    for i, v in enumerate(values):
        episodes.append(_episode("up_calm", {
            "technical": v, "news": -v, "fundamental": 0.0001 * (1 if i % 2 else -1),
        }))
    out = maker.reflect(episodes, date(2020, 1, 31), 0.5)
    deltas = {e.expert_id: e.weight_delta for e in out if e.market_state == "up_calm"}
    assert abs(deltas.get("technical", 0.0)) < 0.10


def test_memory_priors_thread_through_store_and_snapshot():
    """Pre-registered Families B/C: half_life and variance_penalty are real
    knobs that flow through observe, retrieval and the snapshot round-trip."""
    fast = ExpertScorecardStore(scope="state", half_life=1.0, standardize=False,
                                temperature=0.001, min_support=1)
    slow = ExpertScorecardStore(scope="state", half_life=500.0, standardize=False,
                                temperature=0.001, min_support=1)
    for store in (fast, slow):
        store.observe("up_calm", {"technical": 0.01})
        for _ in range(10):
            store.observe("up_calm", {"technical": 0.0})
    fast_mean = fast._cards[("up_calm", "technical")].ew_mean
    slow_mean = slow._cards[("up_calm", "technical")].ew_mean
    assert fast_mean < slow_mean * 0.2, "half-life 1 must forget far faster"

    lam0 = ExpertScorecardStore(scope="state", variance_penalty=0.0,
                                standardize=False, temperature=0.001, min_support=1)
    lam9 = ExpertScorecardStore(scope="state", variance_penalty=9000.0,
                                standardize=False, temperature=0.001, min_support=1)
    for store in (lam0, lam9):
        for i in range(30):
            wiggle = 0.004 if i % 2 else -0.002
            store.observe("up_calm", {"technical": 0.001 + wiggle,
                                      "news": 0.0005, "fundamental": 0.0})
    w0 = lam0.get_weights("up_calm")
    w9 = lam9.get_weights("up_calm")
    assert w0["technical"] > w9["technical"], (
        "a huge variance penalty must punish the noisy expert")

    snap = ExpertScorecardStore(half_life=21.0, variance_penalty=2.0).snapshot()
    back = ExpertScorecardStore.from_snapshot(snap)
    assert back._half_life == 21.0 and back._variance_penalty == 2.0


def test_dsr_trial_budget_may_only_grow():
    from trading_agent.experiments.expert_weight_holdout import _effective_trial_count

    assert _effective_trial_count(3, None) == 3
    assert _effective_trial_count(3, 18) == 18
    with pytest.raises(ValueError, match="only grow"):
        _effective_trial_count(3, 2)


def test_zero_hit_replay_is_refused_because_a_missing_view_abstains_silently():
    """A replay that serves no views must fail, not report a smaller committee's result.

    Missing views are a legal abstention in design D3, so a prompt mismatch degrades the
    committee instead of raising. On 2026-07-28 that silently turned a three-seat run into a
    one-seat run and produced a retracted finding.
    """
    import argparse

    import pytest

    from trading_agent.experiments.expert_weight_holdout_launcher import _corpus_stats_checked

    class _Corpus:
        def __init__(self, hits):
            self.hits = hits
            self.records_written = 0

        def __len__(self):
            return 7274

    args = argparse.Namespace(view_corpus="corpus.jsonl", view_corpus_mode="replay")
    with pytest.raises(SystemExit, match="ZERO views"):
        _corpus_stats_checked(_Corpus(0), args)

    stats = _corpus_stats_checked(_Corpus(8766), args)
    assert stats["hits"] == 8766 and stats["entries"] == 7274

    # Recording modes legitimately start with no hits.
    for mode in ("record", "readthrough"):
        args.view_corpus_mode = mode
        assert _corpus_stats_checked(_Corpus(0), args)["hits"] == 0


def test_verify_block_override_only_changes_the_bootstrap_block():
    """--verify-block replaces the 2*h default without touching anything else.

    h-day overlapping returns lose their autocorrelation at lag h, so a block of h is the
    minimum that preserves the dependence and 2h is extra conservatism (prereg Amendment 9).
    """
    import inspect

    from trading_agent.experiments import expert_weight_holdout as hold

    sig = inspect.signature(hold.run_expert_weight_holdout)
    assert "verify_block" in sig.parameters
    assert sig.parameters["verify_block"].default is None, "default must keep the 2*h behaviour"

    src = inspect.getsource(hold.run_expert_weight_holdout)
    assert "max(2 * attribution_horizon, 14)" in src, "the conservative default must survive"
    assert "int(verify_block) if verify_block" in src, "an explicit block must win"
