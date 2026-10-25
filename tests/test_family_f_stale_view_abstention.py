"""Family F (prereg Amendment 4): stale-view abstention.

An expert whose direction has repeated for more than K active dates leaves the
VOTE until it changes. The rule exists because the fundamental expert's evidence
refreshes 1-4 times a year while it is asked to vote 252 times a year, so in the
test window it becomes a constant (0.056 bits) that still tilts A = sum w*v.

The load-bearing invariant is that the filter touches the vote only: attribution
records read each expert's own direction and must be unchanged, or the trained
memory would stop being comparable to the sealed runs.
"""

import pytest

from trading_agent.contracts import ExpertView, is_safe_default_view
from trading_agent.runner import _apply_stale_view_abstention

TICKERS = ("AAA",)


def _view(expert_id: str, direction: str, *, quality: str = "ok") -> ExpertView:
    return ExpertView(
        expert_id=expert_id,
        direction=direction,
        confidence=0.6,
        reasoning="test",
        data_quality=quality,
    )


def _run(directions_per_date, *, window: int, tickers=TICKERS):
    """Feed a sequence of {expert: direction} dicts; return the kept experts per date."""
    runs: dict[tuple[str, str], tuple[str, int]] = {}
    kept_per_date = []
    for per_expert in directions_per_date:
        views = {
            e: {t: _view(e, d) for t in tickers}
            for e, d in per_expert.items()
        }
        filtered = _apply_stale_view_abstention(views, runs, tickers, window=window)
        kept_per_date.append(set(filtered))
    return kept_per_date


def test_a_view_speaks_for_exactly_window_dates_then_abstains():
    kept = _run([{"technical": "long"}] * 5, window=3)
    assert kept == [{"technical"}, {"technical"}, {"technical"}, set(), set()]


def test_a_changed_direction_restarts_the_run():
    dirs = [{"technical": d} for d in ("long", "long", "long", "flat", "flat", "flat", "flat")]
    kept = _run(dirs, window=2)
    # long speaks twice then mutes; the flip to flat restarts and speaks twice more.
    assert kept == [{"technical"}, {"technical"}, set(),
                    {"technical"}, {"technical"}, set(), set()]


def test_alternating_directions_never_mute():
    dirs = [{"technical": "long" if i % 2 == 0 else "flat"} for i in range(8)]
    assert _run(dirs, window=1) == [{"technical"}] * 8


def test_a_constant_expert_mutes_while_an_informative_one_keeps_voting():
    dirs = [
        {"fundamental": "flat", "technical": "long" if i % 3 else "flat"}
        for i in range(9)
    ]
    kept = _run(dirs, window=2)
    assert kept[0] == {"fundamental", "technical"}
    # fundamental is silent from date 3 onward; technical keeps speaking whenever it moves.
    assert all("fundamental" not in k for k in kept[2:])
    assert any("technical" in k for k in kept[2:])


def test_runs_are_tracked_per_ticker_independently():
    tickers = ("AAA", "BBB")
    runs: dict[tuple[str, str], tuple[str, int]] = {}
    for step in range(4):
        views = {
            "technical": {
                "AAA": _view("technical", "long"),                       # constant
                "BBB": _view("technical", "long" if step % 2 == 0 else "flat"),
            }
        }
        filtered = _apply_stale_view_abstention(views, runs, tickers, window=2)
        kept = filtered.get("technical", {})
        if step >= 2:
            assert "AAA" not in kept, "the constant name must mute"
            assert "BBB" in kept, "the alternating name must keep voting"


def test_abstaining_views_pass_through_untouched():
    """A neutral or safe-default view is already out of the vote; the filter must not
    re-decide it, and its repetition must not mute a later real judgment."""
    runs: dict[tuple[str, str], tuple[str, int]] = {}
    for _ in range(5):
        views = {"news": {"AAA": _view("news", "neutral")}}
        filtered = _apply_stale_view_abstention(views, runs, TICKERS, window=2)
        assert "AAA" in filtered["news"], "neutral views are handed to the aggregation as-is"

    unavailable = ExpertView(
        expert_id="news", direction="flat", confidence=0.0,
        reasoning="data_unavailable", risk_flags=("data_unavailable",),
        data_quality="low",
    )
    assert is_safe_default_view(unavailable), "fixture must be a real safe default"
    runs2: dict[tuple[str, str], tuple[str, int]] = {}
    for _ in range(5):
        filtered = _apply_stale_view_abstention(
            {"news": {"AAA": unavailable}}, runs2, TICKERS, window=2
        )
        assert "AAA" in filtered["news"], "safe defaults are handed through, not double-muted"


def test_an_expert_with_no_surviving_view_is_absent_not_empty():
    kept = _run([{"fundamental": "flat"}] * 4, window=1)
    assert kept[-1] == set(), "a fully muted expert leaves the mapping entirely"


def test_missing_views_are_skipped_without_touching_the_run_length():
    runs: dict[tuple[str, str], tuple[str, int]] = {}
    for present in (True, False, True, True):
        views = {"technical": ({"AAA": _view("technical", "long")} if present else {})}
        _apply_stale_view_abstention(views, runs, TICKERS, window=10)
    assert runs[("technical", "AAA")] == ("long", 3), "only dates with a view advance the run"


@pytest.mark.parametrize("window", [1, 21, 63])
def test_window_is_the_only_difference_between_the_registered_variants(window):
    kept = _run([{"technical": "long"}] * (window + 2), window=window)
    assert kept[:window] == [{"technical"}] * window
    assert kept[window:] == [set(), set()]
