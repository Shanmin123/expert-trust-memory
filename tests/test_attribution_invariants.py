"""Invariants for the counterfactual attribution and its market-adjusted E2 ablation."""

from __future__ import annotations

from datetime import date

import pytest

from trading_agent.contracts import ExpertView
from trading_agent.runner import _attribute


class _FakePanel:
    """Prices for two dates: decision date and the next (forward) date."""

    def __init__(self, p0: dict, p1: dict):
        self._p0 = p0
        self._p1 = p1

    def _price_on(self, ticker, t):
        return self._p0.get(ticker)

    def price_after(self, ticker, t, horizon):
        return self._p1.get(ticker)


class _Obs:
    def __init__(self, tickers, d):
        self.tickers = tickers
        self.decision_date = d


def _view(direction="long", conf=0.6):
    return ExpertView(
        expert_id="technical", direction=direction, confidence=conf,
        reasoning="x", evidence=(), risk_flags=(), data_quality="medium",
    )


def _scored_returns(episodes):
    out = {}
    for ep in episodes:
        for rec in ep.expert_errors:
            out[ep.ticker] = rec.realized_next_return
    return out


# Forward returns +0.20 / +0.10 / 0.00 -> proxy (mean) = +0.10.
_P0 = {"A": 100.0, "B": 100.0, "C": 100.0}
_P1 = {"A": 120.0, "B": 110.0, "C": 100.0}
_D = date(2022, 1, 3)


def _run(mode):
    panel = _FakePanel(_P0, _P1)
    obs = _Obs(("A", "B", "C"), _D)
    views = {"technical": {tk: _view() for tk in obs.tickers}}
    return _attribute(_D, "up_stress", views, {"technical": 1.0}, obs, panel,
                      {tk: "long" for tk in obs.tickers}, attribution_mode=mode)


def test_market_adjusted_excess_sums_to_zero_and_market_is_the_proxy():
    episodes = _run("market_adjusted")
    # market component is the equal-weight proxy (mean forward return), identical per name.
    markets = {round(ep.attribution["market"], 9) for ep in episodes}
    assert markets == {0.10}
    excess = _scored_returns(episodes)
    assert abs(sum(excess.values())) < 1e-9           # excess sums to zero
    assert abs(excess["A"] - 0.10) < 1e-9             # 0.20 - 0.10 proxy
    assert abs(excess["B"] - 0.00) < 1e-9
    assert abs(excess["C"] + 0.10) < 1e-9


def test_external_market_return_is_the_baseline_for_single_name():
    # A single-name universe: without an external market the excess would be
    # realized - realized = 0 (degenerate). With an external market return the
    # baseline is decoupled from the traded stock.
    d = _D
    panel = _FakePanel({"A": 100.0}, {"A": 115.0})   # stock +0.15
    obs = _Obs(("A",), d)
    views = {"technical": {"A": _view()}}

    degenerate = _attribute(d, "up_stress", views, {"technical": 1.0}, obs, panel,
                            {"A": "long"}, attribution_mode="market_adjusted")
    assert _scored_returns(degenerate)["A"] == pytest.approx(0.0)   # excess over itself

    with_market = _attribute(d, "up_stress", views, {"technical": 1.0}, obs, panel,
                             {"A": "long"}, attribution_mode="market_adjusted",
                             market_return=0.10)   # market +0.10
    assert _scored_returns(with_market)["A"] == pytest.approx(0.05)  # 0.15 - 0.10
    assert with_market[0].attribution["market"] == pytest.approx(0.10)


def test_raw_mode_scores_on_raw_return_and_does_not_sum_to_zero():
    episodes = _run("raw")
    raw = _scored_returns(episodes)
    assert abs(raw["A"] - 0.20) < 1e-9
    assert abs(raw["B"] - 0.10) < 1e-9
    assert abs(raw["C"] - 0.00) < 1e-9
    assert abs(sum(raw.values()) - 0.30) < 1e-9       # raw returns, not excess
    # The reporting market component is the shared proxy regardless of learning mode;
    # only the learning signal (scored return above) differs between raw and adjusted.
    assert {round(ep.attribution["market"], 9) for ep in episodes} == {0.10}
    assert episodes[0].attribution["market"] + episodes[0].attribution["signal"] == pytest.approx(
        _scored_returns(_run("raw"))[episodes[0].ticker]
    )  # market + selection reconstruct the realized return


def test_signal_attribution_is_the_signed_vote_times_return():
    # Contribution g = v * r, v = +1 for a long view. System FLAT and a tiny ensemble weight:
    # the contribution still credits the expert's OWN long view on the raw return, independent
    # of the ensemble weight and the system action. Confidence is not used, and no cost enters.
    panel = _FakePanel({"A": 100.0}, {"A": 120.0})   # +0.20
    obs = _Obs(("A",), _D)
    views = {"technical": {"A": _view(direction="long", conf=0.6)}}
    eps = _attribute(_D, "up_stress", views, {"technical": 0.05}, obs, panel,
                     {"A": "flat"}, attribution_mode="raw")
    assert eps[0].signal_attribution["technical"] == pytest.approx(1.0 * 0.20)   # v=+1, not conf
    assert eps[0].action == "flat"                    # system did not act; expert still scored


def test_flat_view_scores_minus_the_return():
    # A flat view is v = -1: in a rising market its contribution is negative (it should have
    # been long). No transaction cost enters the per-expert signal.
    panel = _FakePanel({"A": 100.0}, {"A": 120.0})   # +0.20
    obs = _Obs(("A",), _D)
    views = {"technical": {"A": _view(direction="flat")}}
    eps = _attribute(_D, "up_stress", views, {"technical": 1.0}, obs, panel,
                     {"A": "flat"}, attribution_mode="raw")
    assert eps[0].signal_attribution["technical"] == pytest.approx(-1.0 * 0.20)


def test_safe_default_view_is_excluded_from_learning_evidence():
    # A safe-default flat (data unavailable / parse error) is not a judgment: it must not
    # enter the contribution scorecard, or an expert would be punished for missing data
    # (e.g. the news expert over the years before news coverage starts).
    from trading_agent.experts.base_expert import _safe_default, is_safe_default_view

    panel = _FakePanel({"A": 100.0}, {"A": 120.0})   # +0.20
    obs = _Obs(("A",), _D)
    default_view = _safe_default("news", "data_unavailable")
    assert is_safe_default_view(default_view)
    views = {
        "technical": {"A": _view(direction="long")},
        "news": {"A": default_view},
    }
    eps = _attribute(_D, "up_stress", views, {"technical": 0.5, "news": 0.5}, obs, panel,
                     {"A": "long"}, attribution_mode="raw")
    assert "news" not in eps[0].signal_attribution           # no learning evidence
    assert eps[0].signal_attribution["technical"] == pytest.approx(0.20)


def test_genuine_flat_view_still_counts_as_learning_evidence():
    # Only the safe DEFAULT is excluded; a real flat judgment keeps its v = -1 channel
    # (the down-state usefulness of flat views is a designed learning signal).
    panel = _FakePanel({"A": 100.0}, {"A": 120.0})
    obs = _Obs(("A",), _D)
    genuine_flat = ExpertView(
        expert_id="news", direction="flat", confidence=0.55,
        reasoning="negative tone in filings", evidence=("e",), risk_flags=(),
        data_quality="medium",
    )
    views = {"news": {"A": genuine_flat}}
    eps = _attribute(_D, "up_stress", views, {"news": 1.0}, obs, panel,
                     {"A": "flat"}, attribution_mode="raw")
    assert eps[0].signal_attribution["news"] == pytest.approx(-0.20)
