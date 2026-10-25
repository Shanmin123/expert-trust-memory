"""Learned expert-weight memory: expert_id x market_state -> weight_delta."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Optional, Sequence

from data_collection.storage import connect_sqlite

from trading_agent.aggregation.soft_weighting import EXPERT_IDS
from trading_agent.aggregation.weights import WEIGHT_FLOOR, project_to_floored_simplex

logger = logging.getLogger(__name__)

MARKET_STATES: tuple[str, ...] = ("up_calm", "up_stress", "down_calm", "down_stress")
GLOBAL_MARKET_STATE: str = "global"
MEMORY_STATES: tuple[str, ...] = MARKET_STATES + (GLOBAL_MARKET_STATE,)
MEMORY_STATUSES: tuple[str, ...] = ("learned", "deprecated")

WEIGHT_DELTA_MIN: float = -0.10
WEIGHT_DELTA_MAX: float = 0.10

_EXPERT_ID_SET = frozenset(EXPERT_IDS)
_MEMORY_STATE_SET = frozenset(MEMORY_STATES)
_STATUS_SET = frozenset(MEMORY_STATUSES)


def equal_weights() -> dict[str, float]:
    """The equal-weight baseline: 0.25 per expert."""
    return {eid: 1.0 / len(EXPERT_IDS) for eid in EXPERT_IDS}


def clip_weight_delta(
    value: float,
    *,
    expert_id: Optional[str] = None,
    market_state: Optional[str] = None,
) -> float:
    """Clip a proposed delta to the allowed range and warn when clipping occurs."""
    v = float(value)
    if v < WEIGHT_DELTA_MIN:
        logger.warning(
            "weight_delta %.6f for (expert=%s, market_state=%s) below %.2f; clipped to %.2f",
            v, expert_id, market_state, WEIGHT_DELTA_MIN, WEIGHT_DELTA_MIN,
        )
        return WEIGHT_DELTA_MIN
    if v > WEIGHT_DELTA_MAX:
        logger.warning(
            "weight_delta %.6f for (expert=%s, market_state=%s) above %.2f; clipped to %.2f",
            v, expert_id, market_state, WEIGHT_DELTA_MAX, WEIGHT_DELTA_MAX,
        )
        return WEIGHT_DELTA_MAX
    return v


@dataclass(frozen=True)
class ExpertWeightEntry:
    """One learned weight adjustment for an expert under a market state."""

    entry_id: str
    expert_id: str
    market_state: str
    weight_delta: float
    status: str = "learned"
    created_at: Optional[Date] = None
    learned_at: Optional[Date] = None
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.expert_id not in _EXPERT_ID_SET:
            raise ValueError(f"unknown expert_id {self.expert_id!r}; must be one of {EXPERT_IDS}")
        if self.market_state not in _MEMORY_STATE_SET:
            raise ValueError(
                f"unknown market_state {self.market_state!r}; must be one of {MEMORY_STATES}"
            )
        if self.status not in _STATUS_SET:
            raise ValueError(f"unknown status {self.status!r}; must be one of {MEMORY_STATUSES}")
        clipped = clip_weight_delta(
            self.weight_delta, expert_id=self.expert_id, market_state=self.market_state
        )
        if clipped != self.weight_delta:
            object.__setattr__(self, "weight_delta", clipped)


def _clamp_delta(value: float) -> float:
    """Silent clamp for cumulative deltas (per-entry clipping already warned)."""
    return max(WEIGHT_DELTA_MIN, min(WEIGHT_DELTA_MAX, value))


def get_weights(
    market_state: str,
    learned_entries: Sequence[ExpertWeightEntry] = (),
    *,
    include_global: bool = True,
    clip_cumulative: bool = True,
) -> dict[str, float]:
    """Compute floored, normalised expert weights for a market state.

    Entries accumulate additively per (expert_id, memory-state key). By default
    the SUMMED delta per key is clipped to the same [-0.10, 0.10] bound as a
    single entry before projection, so repeated reflection rounds cannot push
    the effective adjustment beyond the documented bound. ``clip_cumulative=False``
    disables that clip and is provided only for the bounded-vs-unbounded ablation
    (E3); the main configuration always clips.
    """
    if market_state not in MARKET_STATES:
        raise ValueError(
            f"unknown market_state {market_state!r}; must be one of {MARKET_STATES}"
        )
    bound = _clamp_delta if clip_cumulative else (lambda v: v)
    state_delta: dict[str, float] = {}
    global_delta: dict[str, float] = {}
    for entry in learned_entries:
        if entry.status != "learned":
            continue
        if entry.market_state == market_state:
            state_delta[entry.expert_id] = state_delta.get(entry.expert_id, 0.0) + entry.weight_delta
        elif include_global and entry.market_state == GLOBAL_MARKET_STATE:
            global_delta[entry.expert_id] = global_delta.get(entry.expert_id, 0.0) + entry.weight_delta
    weights = equal_weights()
    for expert_id in set(state_delta) | set(global_delta):
        weights[expert_id] = (
            weights.get(expert_id, 0.0)
            + bound(state_delta.get(expert_id, 0.0))
            + bound(global_delta.get(expert_id, 0.0))
        )
    return project_to_floored_simplex(weights, WEIGHT_FLOOR)


_DDL = """
CREATE TABLE IF NOT EXISTS expert_weight_entries (
    entry_id    TEXT PRIMARY KEY,
    expert_id   TEXT NOT NULL,
    market_state TEXT NOT NULL,
    weight_delta REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'learned',
    created_at  TEXT,
    learned_at  TEXT,
    rationale   TEXT
);
CREATE INDEX IF NOT EXISTS idx_expert_weights_state_status
    ON expert_weight_entries (market_state, status);
"""


def _d(value: Optional[Date]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _pd(value: Optional[str]) -> Optional[Date]:
    if not value:
        return None
    if isinstance(value, Date):
        return value
    text = str(value)
    try:
        return Date.fromisoformat(text[:10])
    except ValueError:
        import pandas as pd

        return pd.to_datetime(value).date()


class ExpertWeightStore:
    """SQLite-backed store for learned expert-weight adjustments."""

    def __init__(self, path: str = ":memory:", *, clip_cumulative: bool = True) -> None:
        self._conn = connect_sqlite(path)
        self._conn.executescript(_DDL)
        self._conn.commit()
        self._frozen_at: Optional[Date] = None
        # Bounding policy: the main configuration clips the summed per-key delta;
        # clip_cumulative=False is the unbounded-memory ablation (E3).
        self._clip_cumulative = bool(clip_cumulative)
        self._learned_cache: Optional[list[ExpertWeightEntry]] = None
        self._weights_cache: dict[tuple[str, bool, bool], dict[str, float]] = {}

    @property
    def frozen_at(self) -> Optional[Date]:
        return self._frozen_at

    def close(self) -> None:
        self._conn.close()

    def freeze(self, as_of: Date) -> None:
        self._frozen_at = as_of

    def _ensure_writable(self) -> None:
        if self._frozen_at is not None:
            raise RuntimeError(f"ExpertWeightStore is frozen as of {_d(self._frozen_at)}")

    def _invalidate_cache(self) -> None:
        self._learned_cache = None
        self._weights_cache.clear()

    def add_learned_adjustment(self, entry: ExpertWeightEntry) -> str:
        """Persist one learned adjustment."""
        self._ensure_writable()
        if entry.status != "learned":
            raise ValueError("only learned entries can be written as active adjustments")
        clipped = clip_weight_delta(
            entry.weight_delta, expert_id=entry.expert_id, market_state=entry.market_state
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO expert_weight_entries
              (entry_id, expert_id, market_state, weight_delta, status,
               created_at, learned_at, rationale)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                entry.entry_id,
                entry.expert_id,
                entry.market_state,
                clipped,
                "learned",
                _d(entry.created_at),
                _d(entry.learned_at),
                entry.rationale,
            ),
        )
        self._conn.commit()
        self._invalidate_cache()
        return entry.entry_id

    def deprecate(self, entry_id: str) -> None:
        self._ensure_writable()
        self._conn.execute(
            "UPDATE expert_weight_entries SET status = 'deprecated' WHERE entry_id = ?",
            (entry_id,),
        )
        self._conn.commit()
        self._invalidate_cache()

    def _row_to_entry(self, row) -> ExpertWeightEntry:
        entry_id, expert_id, market_state, weight_delta, status, created, learned, rationale = row
        return ExpertWeightEntry(
            entry_id=entry_id,
            expert_id=expert_id,
            market_state=market_state,
            weight_delta=weight_delta,
            status=status,
            created_at=_pd(created),
            learned_at=_pd(learned),
            rationale=rationale or "",
        )

    def _select(self, where: str = "", params: Sequence = ()) -> list[ExpertWeightEntry]:
        sql = (
            "SELECT entry_id, expert_id, market_state, weight_delta, status, "
            "created_at, learned_at, rationale "
            "FROM expert_weight_entries " + where
        )
        return [self._row_to_entry(r) for r in self._conn.execute(sql, params).fetchall()]

    def all_entries(self) -> list[ExpertWeightEntry]:
        return self._select()

    def learned(self) -> list[ExpertWeightEntry]:
        if self._learned_cache is None:
            self._learned_cache = self._select("WHERE status = 'learned'")
        return list(self._learned_cache)

    def deprecated(self) -> list[ExpertWeightEntry]:
        return self._select("WHERE status = 'deprecated'")

    def count_learned(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM expert_weight_entries WHERE status = 'learned'"
        ).fetchone()
        return int(row[0])

    def get_weights(
        self,
        market_state: str,
        *,
        include_global: bool = True,
        clip_cumulative: bool | None = None,
    ) -> dict[str, float]:
        """Floored, normalised four-weight dict from learned memory.

        The bounding policy defaults to the store's own (set at construction);
        pass ``clip_cumulative`` to override it for a single call. The cache is
        keyed by the effective policy so a bounded and an unbounded read cannot
        cross-contaminate.
        """
        effective = self._clip_cumulative if clip_cumulative is None else bool(clip_cumulative)
        key = (market_state, include_global, effective)
        cached = self._weights_cache.get(key)
        if cached is not None:
            return dict(cached)
        weights = get_weights(
            market_state, self.learned(), include_global=include_global, clip_cumulative=effective
        )
        self._weights_cache[key] = dict(weights)
        return weights

    def snapshot(self) -> tuple[tuple, ...]:
        rows = self._conn.execute(
            """
            SELECT entry_id, expert_id, market_state, weight_delta, status,
                   created_at, learned_at, rationale
            FROM expert_weight_entries
            ORDER BY entry_id
            """
        ).fetchall()
        return tuple(tuple(r) for r in rows)


def entry_to_dict(entry: ExpertWeightEntry) -> dict:
    return {
        "entry_id": entry.entry_id,
        "expert_id": entry.expert_id,
        "market_state": entry.market_state,
        "weight_delta": float(entry.weight_delta),
        "status": entry.status,
        "created_at": _d(entry.created_at),
        "learned_at": _d(entry.learned_at),
        "rationale": entry.rationale,
    }


def entry_from_dict(payload: dict) -> ExpertWeightEntry:
    return ExpertWeightEntry(
        entry_id=str(payload["entry_id"]),
        expert_id=str(payload["expert_id"]),
        market_state=str(payload["market_state"]),
        weight_delta=float(payload["weight_delta"]),
        status=str(payload.get("status", "learned")),
        created_at=_pd(payload.get("created_at")),
        learned_at=_pd(payload.get("learned_at")),
        rationale=str(payload.get("rationale", "")),
    )


def snapshot_payload(store: ExpertWeightStore, *, status: str = "frozen") -> dict:
    payload = {
        "status": status,
        "frozen_at": _d(store.frozen_at),
        "entries": [entry_to_dict(entry) for entry in store.learned()],
    }
    return payload


def load_snapshot_payload(path: str | Path) -> dict:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "memory_snapshot" in payload:
        return payload["memory_snapshot"]
    return payload


def snapshot_entries(payload: dict) -> list[ExpertWeightEntry]:
    return [entry_from_dict(item) for item in payload.get("entries", [])]
