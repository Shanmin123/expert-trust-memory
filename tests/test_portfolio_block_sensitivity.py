"""The portfolio bootstrap must answer the block-length question reproducibly.

The thesis reports that the portfolio p-value moves between .074 and .098 across block
lengths 7, 21 and 42. That claim had no committed script behind it, so it could not be
re-derived. These tests pin the statistic and the properties a reader would check.
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "portfolio_block_sensitivity",
    Path(__file__).resolve().parents[1] / "scripts" / "portfolio_block_sensitivity.py",
)
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)


def test_t_statistic_matches_the_textbook_definition():
    mean, se, t = mod.t_stat([1.0, 2.0, 3.0, 4.0])
    assert mean == pytest.approx(2.5)
    assert se == pytest.approx((1.2909944487358056) / 2.0)   # sd / sqrt(n)
    assert t == pytest.approx(mean / se)


def test_a_series_centred_on_zero_is_not_significant():
    rng = random.Random(7)
    diffs = [rng.gauss(0.0, 1e-4) for _ in range(352)]
    out = mod.studentised_p(diffs, block=21, n_boot=400, seed=1)
    assert out["p_one_sided"] > 0.10


def test_a_strongly_positive_series_is_significant():
    rng = random.Random(7)
    diffs = [rng.gauss(5e-4, 1e-4) for _ in range(352)]
    out = mod.studentised_p(diffs, block=21, n_boot=400, seed=1)
    assert out["p_one_sided"] < 0.02


def test_p_values_use_the_plus_one_convention_so_zero_is_impossible():
    diffs = [1e-3] * 200            # no variation once resampled, degenerate on purpose
    out = mod.studentised_p(diffs, block=7, n_boot=100, seed=1)
    assert out["p_one_sided"] >= 1.0 / 101
    assert out["p_two_sided"] >= 1.0 / 101


def test_the_two_sided_p_is_never_below_the_one_sided_p():
    rng = random.Random(3)
    diffs = [rng.gauss(1e-4, 1e-4) for _ in range(352)]
    for block in mod.BLOCKS:
        out = mod.studentised_p(diffs, block=block, n_boot=300, seed=5)
        assert out["p_two_sided"] >= out["p_one_sided"]


def test_the_same_seed_reproduces_the_same_p():
    rng = random.Random(11)
    diffs = [rng.gauss(1e-4, 1e-4) for _ in range(352)]
    a = mod.studentised_p(diffs, block=21, n_boot=300, seed=99)
    b = mod.studentised_p(diffs, block=21, n_boot=300, seed=99)
    assert a == b


def test_the_mean_reported_is_the_observed_one_not_a_resampled_one():
    diffs = [0.001, 0.002, 0.003]
    out = mod.studentised_p(diffs, block=2, n_boot=50, seed=1)
    assert out["mean_daily_diff"] == pytest.approx(0.002)


def test_blocks_cover_the_default_the_horizon_and_twice_the_horizon():
    # 7 is the cube-root default, 21 the attribution horizon, 42 what cells use.
    assert mod.BLOCKS == (7, 21, 42)
