"""The released memory snapshots are the ones the dissertation fixed before the live run.

Each snapshot carries a SHA-256 of every other field, so an edit to any field is caught here.
The weights the snapshots deploy are pinned to the table the dissertation registered before
the forward run began: fundamental, news and technical, retrieved at the selected
temperature of 0.5.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trading_agent.memory.expert_scorecard import ExpertScorecardStore

MEMORY = Path(__file__).resolve().parents[1] / "frozen_memory"
TAU = 0.5
ORDER = ("fundamental", "news", "technical")
EQUAL = (0.333, 0.333, 0.333)

DRAW = {"ADSK", "AES", "AFL", "AVY", "AZO", "BBY", "BIIB", "C", "CTAS", "CVX", "DDS",
        "DHI", "ECL", "INCY", "MTCH", "O", "PNC", "RL", "TEX", "TGT"}

#: stock -> (deployed memory, verified records, weights), as registered before the run.
REGISTERED = {
    "BIIB": ("global", 3, (0.415, 0.346, 0.239)),
    "ECL": ("global", 2, (0.566, 0.228, 0.206)),
    "AZO": ("global", 2, (0.573, 0.341, 0.087)),
    "CTAS": ("global", 3, (0.635, 0.105, 0.260)),
    "DDS": ("global", 1, (0.369, 0.262, 0.369)),
    "INCY": ("global", 1, (0.371, 0.258, 0.371)),
    "AES": ("global", 1, (0.385, 0.229, 0.385)),
}
NOTHING_VERIFIED = ("AFL", "AVY", "BBY", "C", "CVX", "DHI", "MTCH", "O", "PNC", "RL", "TEX", "TGT")


def _load(name: str) -> dict:
    return json.loads((MEMORY / name).read_text(encoding="utf-8"))


def _content_hash(payload: dict) -> str:
    """SHA-256 of the canonical JSON of every field except the hash itself."""
    body = {k: v for k, v in payload.items() if k != "hash_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _weights(payload: dict, state: str) -> tuple[float, ...]:
    store = ExpertScorecardStore.from_snapshot(payload, temperature=TAU)
    w = store.get_weights(state)
    return tuple(round(w[e], 3) for e in ORDER)


SNAPSHOTS = sorted(p.name for p in MEMORY.glob("*.json"))


def test_the_release_is_the_twenty_stock_draw():
    global_stocks = {n.split("__")[0] for n in SNAPSHOTS if n.endswith("__global.json")}
    state_stocks = {n.split("__")[0] for n in SNAPSHOTS if n.endswith("__state.json")}
    assert global_stocks == DRAW
    assert state_stocks <= DRAW
    assert "ADSK" in state_stocks, "ADSK is the one stock deployed on its state memory"
    assert len(SNAPSHOTS) == len(global_stocks) + len(state_stocks)


@pytest.mark.parametrize("name", SNAPSHOTS)
def test_every_snapshot_hash_verifies(name):
    payload = _load(name)
    assert payload["hash_sha256"] == _content_hash(payload)


def test_the_hash_detects_an_edit():
    payload = _load("AZO__global.json")
    payload["temperature"] = 0.001
    assert payload["hash_sha256"] != _content_hash(payload)


@pytest.mark.parametrize("name", SNAPSHOTS)
def test_every_snapshot_is_frozen_at_the_end_of_training(name):
    payload = _load(name)
    assert payload["status"] == "frozen"
    assert payload["frozen"] is True
    assert payload["frozen_at"] == "2023-12-31"
    assert payload["temperature"] == TAU


@pytest.mark.parametrize("stock", sorted(REGISTERED))
def test_deployed_weights_match_the_registered_table(stock):
    scope, records, want = REGISTERED[stock]
    payload = _load(f"{stock}__{scope}.json")
    assert len(payload.get("certified_cells") or []) == records
    assert _weights(payload, "global") == want


def test_adsk_leaves_equal_weights_only_in_down_stress():
    payload = _load("ADSK__state.json")
    assert len(payload.get("certified_cells") or []) == 1
    assert _weights(payload, "down_stress") == (0.456, 0.456, 0.088)
    for state in ("up_calm", "up_stress", "down_calm"):
        assert _weights(payload, state) == EQUAL, state


@pytest.mark.parametrize("stock", NOTHING_VERIFIED)
def test_a_stock_with_nothing_verified_deploys_equal_weights(stock):
    payload = _load(f"{stock}__global.json")
    assert not (payload.get("certified_cells") or [])
    assert _weights(payload, "global") == EQUAL
