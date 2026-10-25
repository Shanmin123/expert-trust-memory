"""Family E (prereg Amendment 3) decision-layer tests.

E1: hit-rate log-odds retrieval (``score_mode="hit_logodds"``) — ordering,
Laplace shrinkage, certified-only gating, snapshot round-trip, delta on the
logit scale, and the sealed default staying ``ewvar``.

E2: proportional sizing in the deterministic risk layer — the binary default
must stay byte-identical, and the fraction path must target, band, and unwind
correctly. Entry/exit dates are owned by the direction layer in both modes.
"""

import pytest

from trading_agent.contracts import ALL_AVAILABLE
from trading_agent.memory.expert_scorecard import (
    EXPERT_IDS,
    ExpertScorecardStore,
    equal_weights,
)
from trading_agent.runner import REBALANCE_BAND, _direction_to_intents

STATE = "up_calm"


def _feed(store: ExpertScorecardStore, hit_rates: dict[str, float], n: int = 40) -> None:
    """Feed n episodes where each expert wins on a fixed fraction of them."""
    for i in range(n):
        contributions = {
            expert: (0.01 if i < hit_rates[expert] * n else -0.01)
            for expert in EXPERT_IDS
        }
        store.observe(STATE, contributions)


def _ranked(weights: dict[str, float]) -> list[str]:
    return sorted(weights, key=weights.get, reverse=True)


# ---------------------------------------------------------------------------
# E1: hit-rate log-odds retrieval
# ---------------------------------------------------------------------------

def test_hit_logodds_orders_experts_by_hit_rate():
    store = ExpertScorecardStore(scope="state", score_mode="hit_logodds")
    _feed(store, {"technical": 0.75, "news": 0.50, "fundamental": 0.25})
    weights = store.get_weights(STATE)
    assert _ranked(weights) == ["technical", "news", "fundamental"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert min(weights.values()) >= 0.05 - 1e-9


def test_hit_logodds_laplace_shrinks_small_samples_toward_equal():
    small = ExpertScorecardStore(scope="state", score_mode="hit_logodds", min_support=2)
    big = ExpertScorecardStore(scope="state", score_mode="hit_logodds", min_support=2)
    rates = {"technical": 0.75, "news": 0.50, "fundamental": 0.25}
    _feed(small, rates, n=4)
    _feed(big, rates, n=80)
    w_small = small.get_weights(STATE)
    w_big = big.get_weights(STATE)
    third = 1.0 / len(EXPERT_IDS)
    assert abs(w_small["technical"] - third) < abs(w_big["technical"] - third)


def test_hit_logodds_below_min_support_falls_back_to_equal():
    store = ExpertScorecardStore(scope="state", score_mode="hit_logodds")
    _feed(store, {"technical": 1.0, "news": 0.0, "fundamental": 0.0}, n=5)
    assert store.get_weights(STATE) == equal_weights()


def test_hit_logodds_certified_only_gates_retrieval():
    store = ExpertScorecardStore(
        scope="state", score_mode="hit_logodds", certified_only=True
    )
    _feed(store, {"technical": 0.9, "news": 0.5, "fundamental": 0.1})
    assert store.get_weights(STATE) == equal_weights()
    store.set_certified_cells([(STATE, "technical")])
    weights = store.get_weights(STATE)
    assert weights["technical"] > weights["news"]
    assert weights["news"] == pytest.approx(weights["fundamental"])


def test_hit_logodds_snapshot_roundtrip_and_temperature_reread():
    store = ExpertScorecardStore(scope="state", score_mode="hit_logodds")
    _feed(store, {"technical": 0.75, "news": 0.50, "fundamental": 0.25})
    rebuilt = ExpertScorecardStore.from_snapshot(store.snapshot(), temperature=2.0)
    reference = ExpertScorecardStore(
        scope="state", score_mode="hit_logodds", temperature=2.0
    )
    _feed(reference, {"technical": 0.75, "news": 0.50, "fundamental": 0.25})
    for expert in EXPERT_IDS:
        assert rebuilt.get_weights(STATE)[expert] == pytest.approx(
            reference.get_weights(STATE)[expert]
        )
    # A payload predating score_mode must rebuild as the sealed ewvar mode.
    legacy = store.snapshot()
    legacy.pop("score_mode")
    assert ExpertScorecardStore.from_snapshot(legacy)._score_mode == "ewvar"


def test_hit_logodds_reflection_delta_applies_on_logit_scale():
    store = ExpertScorecardStore(scope="state", score_mode="hit_logodds")
    _feed(store, {expert: 0.5 for expert in EXPERT_IDS})
    assert store.get_weights(STATE) == equal_weights()
    store.set_reflection_delta(STATE, "technical", 0.10)
    assert store.get_weights(STATE)["technical"] > 1.0 / len(EXPERT_IDS)
    ignored = ExpertScorecardStore(
        scope="state", score_mode="hit_logodds", use_reflection=False
    )
    _feed(ignored, {expert: 0.5 for expert in EXPERT_IDS})
    ignored.set_reflection_delta(STATE, "technical", 0.10)
    assert ignored.get_weights(STATE) == equal_weights()


def test_score_mode_validation_and_sealed_default():
    assert ExpertScorecardStore(scope="state")._score_mode == "ewvar"
    with pytest.raises(ValueError):
        ExpertScorecardStore(scope="state", score_mode="bogus")
    with pytest.raises(ValueError):
        ExpertScorecardStore(
            scope="state", score_mode="hit_logodds", shrink_to_global=True
        )


# ---------------------------------------------------------------------------
# E2: proportional sizing in the deterministic risk layer
# ---------------------------------------------------------------------------

class _Bar:
    def __init__(self, price: float) -> None:
        self.adjusted_close = price


class _Obs:
    def __init__(self, prices: dict[str, float]) -> None:
        self.tickers = list(prices)
        self.bars = {ticker: _Bar(price) for ticker, price in prices.items()}


class _Account:
    def __init__(self, cash: float) -> None:
        self.cash = cash


class _Pos:
    def __init__(self, quantity: int) -> None:
        self.quantity = quantity


class _Positions:
    def __init__(self, by_ticker: dict[str, _Pos] | None = None) -> None:
        self._by_ticker = dict(by_ticker or {})

    def by_ticker(self) -> dict[str, _Pos]:
        return dict(self._by_ticker)


def test_binary_default_intents_unchanged():
    obs = _Obs({"AAA": 10.0})
    # Entry: full cash budget, no fractions.
    intents = _direction_to_intents(
        {"AAA": "long"}, obs, _Account(10_000.0), _Positions()
    )
    assert [(i.side, i.quantity) for i in intents] == [("buy", 1000)]
    # Held long: no orders.
    assert _direction_to_intents(
        {"AAA": "long"}, obs, _Account(0.0), _Positions({"AAA": _Pos(1000)})
    ) == []
    # Flat: sell everything.
    sells = _direction_to_intents(
        {"AAA": "flat"}, obs, _Account(0.0), _Positions({"AAA": _Pos(1000)})
    )
    assert [(i.side, i.quantity) for i in sells] == [("sell", ALL_AVAILABLE)]


def test_proportional_entry_buys_the_target_fraction():
    obs = _Obs({"AAA": 10.0})
    intents = _direction_to_intents(
        {"AAA": "long"}, obs, _Account(10_000.0), _Positions(),
        fractions_by_ticker={"AAA": 0.5},
    )
    assert [(i.side, i.quantity) for i in intents] == [("buy", 500)]


def test_proportional_band_suppresses_small_rebalances():
    obs = _Obs({"AAA": 10.0})
    held = _Positions({"AAA": _Pos(500)})
    # Sleeve = 5000 cash + 5000 held = 10000; target 0.55 is a 5% move: inside band.
    assert _direction_to_intents(
        {"AAA": "long"}, obs, _Account(5_000.0), held,
        fractions_by_ticker={"AAA": 0.55},
    ) == []
    # Target 0.8 is a 30% move: buy up to 800 shares.
    ups = _direction_to_intents(
        {"AAA": "long"}, obs, _Account(5_000.0), held,
        fractions_by_ticker={"AAA": 0.8},
    )
    assert [(i.side, i.quantity) for i in ups] == [("buy", 300)]
    # Target 0.2: partial sell down to 200 shares.
    downs = _direction_to_intents(
        {"AAA": "long"}, obs, _Account(5_000.0), held,
        fractions_by_ticker={"AAA": 0.2},
    )
    assert [(i.side, i.quantity) for i in downs] == [("sell", 300)]
    assert 0.0 < REBALANCE_BAND < 1.0


def test_proportional_flat_still_sells_everything():
    obs = _Obs({"AAA": 10.0})
    sells = _direction_to_intents(
        {"AAA": "flat"}, obs, _Account(0.0), _Positions({"AAA": _Pos(400)}),
        fractions_by_ticker={"AAA": 0.0},
    )
    assert [(i.side, i.quantity) for i in sells] == [("sell", ALL_AVAILABLE)]
