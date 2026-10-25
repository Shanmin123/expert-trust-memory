"""Test 2: data-layer leakage guard — no Observation carries availability_time > decision_time.

This checks the data layer independently of the execution-time test: every sampled
Observation must satisfy availability_time <= decision_time.
"""

from __future__ import annotations

from trading_agent.execution.finsaber_adapter import FinsaberBacktestAdapter


def test_observation_availability_not_in_future(demo_dataset, demo_config):
    adapter = FinsaberBacktestAdapter(demo_dataset, demo_config)
    sampled = 0
    for d in adapter.dates:
        obs = adapter.get_observation(d)
        assert obs.availability_time <= obs.decision_time, (
            f"look-ahead: availability {obs.availability_time} > decision {obs.decision_time}"
        )
        assert obs.decision_date == d
        sampled += 1
    assert sampled == len(adapter.dates) >= 2
